# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from scipy.signal import resample_poly
from datetime import datetime
from time import sleep
from queue import Queue
from combadge_wavHeader import CombadgeWavHeader as wavHeader
from threading import Thread
import socket
import netifaces
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Generator, Iterable ## ??
from holoscan.core import Application, Operator, OperatorSpec
# from holoscan.conditions import CountCondition
# from riva.client.auth import Auth
# import riva.client
# import riva.client.proto.riva_asr_pb2 as rasr
# import riva.client.proto.riva_asr_pb2_grpc as rasr_srv

class ComBadgeServerRXOp(Operator):
    """
    Wi-Fi server Operator that receives streaming microphone data from an external edge device
    and saves it in a Queue until recording is done. Then, it sends the data chunck from the Queue Riva ASR Operator 
    """
    def __init__(self, fragment, *args, **kwargs):
        # network variables used to receive and send data from server (Operator) to clients (device)
        self.host = kwargs.pop("host", "localhost")
        self.port = kwargs.pop("port", 8080)
        self.server_socket = None
        self.passdown_queue = Queue()
        # self.clients = set()
        super().__init__(fragment, *args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.output("server_rx_response")
        
    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f'Serving on {self.host}:{self.port}')

        listener_thread = Thread(target=self.listen_client)
        listener_thread.start()

    def listen_client(self):
        try:
            while True:
                client_socket, client_address = self.server_socket.accept()
                print(f"Connected by {client_address}")
                client_thread = ClientHandler(client_socket, client_address, self.passdown_queue)
                # self.clients.add(client_thread)
        
        except KeyboardInterrupt:
            print("Server is shutting down...")
        
        finally:
            if self.server_socket:
                self.server_socket.close()

    def compute(self, op_input, op_output, context):
        while not self.passdown_queue.empty():
            server_rx_response = self.passdown_queue.get()
            op_output.emit(server_rx_response, "server_rx_response")
        
class ClientHandler(Thread):
    def __init__(self, conn, addr, passdown_queue):
        Thread.__init__(self)
        self.client_socket = conn
        self.client_address = addr
        self.passdown_queue = passdown_queue
        self.wavHeader = wavHeader()
        self.audio_queue = AudioQueue()
        self.end_of_audio = bytes("END_OF_AUDIO", 'utf-8')
        self.daemon = True # Allow thread to be killed when the main program exists
        self.start()
        
    def run(self):
        print(f'Connected by {self.client_address}')
        
        try:
            # Receive wav header
            is_wavHeader = 0
            while not is_wavHeader:
                wav_bytes = self.client_socket.recv(44)
                if (not wav_bytes):
                    break
                print(f"wav_bytes length: {len(wav_bytes)}")
                # print(f"wav_header: {wav_data.hex()}")
                is_wavHeader = self.wavHeader.set_wavHeader_from_bitstr(wav_bytes, len(wav_bytes))
                if (not is_wavHeader):
                    self.client_socket.send("Retry")
            self.client_socket.send("Ready")

            # Receive audio data
            while True:
                audio_bytes = self.client_socket.recv(1024)
                if not audio_bytes:
                    break
                
                print(f'{len(audio_bytes)} data samples received from {self.client_address}') # : {message.strip()}
                
                # compare data to end_of_audio message
                if (self.end_of_audio in audio_bytes):
                    byte_str = self.audio_queue.get_all()
                    server_rx_response = {
                        "data": byte_str,
                        "client-ip": self.client_address,
                        "sample-rate": self.wavHeader.sample_rate,
                    }
                    self.passdown_queue.put(server_rx_response)
                    # make sure queue is empty
                    if (self.audio_queue.size != 0):
                        print("Error getting audio data from queue")
                
                # add data to queue
                else:
                    self.audio_queue.put(audio_bytes)

        except Exception as e:
            print(f"Error handling client {self.client_address}: {e}")
        
        finally:
            print(f'Lost connection from {self.client_address}')
            self.client_socket.close()

class AudioQueue:
    def __init__(self):
        self.queue = Queue()
        self.size = 0

    def put(self, samples):
        self.queue.put(samples)
        self.size += len(samples)
        
    def get_all(self) -> bytes:
        data = []
        
        while not self.queue.empty():
            data.append(self.queue.get())
            self.size -= len(data[-1])

        return b''.join(data)

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        pass
    
    def __iter__(self):
        return self
        