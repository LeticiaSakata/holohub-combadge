import socket
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.conditions import CountCondition

class SocketServerOp(Operator):
    def __init__(self, *args, **kwargs):
        self.host = kwargs.pop("host", "localhost")
        self.port = kwargs.pop("port", 9999)
        super().__init__(*args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.output("out")
    
    def compute(self, op_input, op_output, context):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print(f"Listening on {self.host}:{self.port}")
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    op_output.emit(data, "out")

class PrintOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")
        print(f"Received: {data.decode('utf-8')}")

class SocketApp(Application):
    def compose(self):
        socket_server = SocketServerOp(self, CountCondition(self, count=1), name="socket_server", host="localhost", port=9999)
        print_op = PrintOp(self, name="print_op")
        self.add_flow(socket_server, print_op)

if __name__ == "__main__":
    app = SocketApp()
    app.config("")
    app.run()
