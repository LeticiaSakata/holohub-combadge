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
        # self.clients = set()
        super().__init__(fragment, *args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.output("server_rx_response")

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f'Serving on {self.host}:{self.port}')
        
    def compute(self, op_input, op_output, context):
        try:
           while True:
                client_socket, client_address = self.server_socket.accept()
                print(f"Connected by {self.client_address}")
                client_thread = ClientHandler(client_socket, client_address, op_output)
                # self.clients.add(client_thread)
        
        except KeyboardInterrupt:
            print("Server is shutting down...")
        
        finally:
            if self.server_socket:
                self.server_socket.close()
    


class AudioQueue:
    """
    Implement same context manager/iterator interfaces as MicrophoneStream (for ASR.process_audio())
    Credit for this code: https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/llamaspeak/asr.py
    """
    def __init__(self, audio_chunk=1024):
        self.queue = Queue()
        self.audio_chunk = audio_chunk

    def put(self, samples):
        self.queue.put(samples)
        
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        pass
        
    def __next__(self) -> bytes:
        data = []
        size = 0
        
        while size <= self.audio_chunk * 2:
            data.append(self.queue.get())
            size += len(data[-1])

        return b''.join(data)
    
    def __iter__(self):
        return self
        
class ClientHandler(Thread):
    def __init__(self, conn, addr, op_output):
        Thread.__init__(self)
        self.client_socket = conn
        self.client_address = addr
        self.op_output = op_output
        self.audio_queue = AudioQueue()
        self.end_of_audio = "END_OF_AUDIO"
        self.daemon = True # Allow thread to be killed when the main program exists
        self.start()

        # audio variables
        self.wavHeader = wavHeader()


        self.is_speaking = False
        self.audio_bytes = bytes()
        self.done_speaking = False # used to determine if the transcript should be sent as complete
        self.sample_rate_hz = cli_args.sample_rate_hz
        self.whisper_response = ""
        self.whisper_pipeline = self._get_whisper_pipeline()

        # queue used to write ASR responses to in a background thread
        self.streaming_queue = Queue()
        
        # flag that gives Riva time to output final prediction
        self.input_queue = AudioQueue()
        
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
            self.sample_rate_hz = wav_bytes.sample_rate

            # Receive audio data
            while True:
                audio_bytes = self.client_socket.recv(1024)
                if not data:
                    break
                
                print(f'{len(audio_bytes)} data samples received from {self.client_address}') # : {message.strip()}
                
                # compare data to end_of_audio
                if (self.end_of_audio in audio_bytes):
                    bit_str = queue_data
                    server_rx_response = {
                        "data": bit_str,
                        "client-ip": self.client_address,
                    }
                    self.op_output.emit(server_rx_response, "server_rx_response")
                    # clear queue
                    # make sure queue is empty

                # add data to queue
                else:
                    self.audio_queue.put(audio_bytes)

                
        except Exception as e:
            print(f"Error handling client {self.client_address}: {e}")
        
        finally:
            print(f'Lost connection from {self.client_address}')
            self.client_socket.close()





        if not self.streaming_queue.empty():
            # Gather Riva responses
            riva_response = self.streaming_queue.get(True, 0.1)
        else:
            # Sleep if no response, otherwise the app is unable to handle the
            # rate of compute() calls
            sleep(0.05)
            riva_response = None

        whisper_response = self.whisper_response
        if whisper_response:
            self.done_speaking = True
            self.whisper_response = ""
 
        asr_response = {
            "is_speaking": self.is_speaking,
            "done_speaking": self.done_speaking,
            "riva_response": riva_response,
            "whisper_response": whisper_response
        }
        op_output.emit(asr_response, "asr_response")
        # Reset done_speaking flag
        self.done_speaking = False