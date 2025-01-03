#Need to make desktop app with GUI that loads image, user can choose what conversion he wants to make
#Send file to server app with request (via JSON maybe)
#Wait for result, show the result, save result as image.
import socket
import sys

if __name__ == "__main__":
    HOST, PORT = "localhost", 8080
    data = " ".join(sys.argv[1:])

    # Create a socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((HOST, PORT))
        sock.sendall(bytes(data + "\n", "utf-8"))

        # Receive data from the server and shut down
        received = str(sock.recv(1024), "utf-8")

    print("Sent:     {}".format(data))
    print("Received: {}".format(received))