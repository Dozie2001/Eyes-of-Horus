#!/usr/bin/env python3
"""
Diagnose network and find camera.
Run while connected to camera's WiFi hotspot.
"""

import subprocess
import socket
import time


def run(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


print("=" * 50)
print("Network Diagnostic")
print("=" * 50)

# What WiFi are we on?
wifi_name = run("/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -I | awk '/ SSID/ {print substr($0, index($0, $2))}'")
print(f"WiFi network: {wifi_name or 'unknown'}")

# Our IP
local_ip = run("ipconfig getifaddr en0")
print(f"Mac IP:       {local_ip or 'no IP assigned'}")

# Gateway
gateway = run("route -n get default 2>/dev/null | grep gateway | awk '{print $2}'")
print(f"Gateway:      {gateway or 'none'}")

# Subnet mask
netmask = run("ifconfig en0 | grep 'inet ' | awk '{print $4}'")
print(f"Netmask:      {netmask or 'unknown'}")

if not local_ip:
    print("\nNo IP address! WiFi might still be connecting. Wait a few seconds and retry.")
    exit(1)

# Ping gateway
print(f"\n--- Pinging gateway {gateway} ---")
ping_result = run(f"ping -c 2 -W 2 {gateway}")
if "bytes from" in ping_result:
    print(f"Gateway is reachable!")
else:
    print(f"Gateway NOT reachable. Output: {ping_result[:200]}")

# Ping 192.168.169.1 specifically
if gateway != "192.168.169.1":
    print(f"\n--- Pinging 192.168.169.1 (expected camera IP) ---")
    ping2 = run("ping -c 2 -W 2 192.168.169.1")
    if "bytes from" in ping2:
        print("192.168.169.1 is reachable!")
    else:
        print("192.168.169.1 NOT reachable")

# Scan ports on gateway
target = gateway or "192.168.169.1"
print(f"\n--- Scanning ports on {target} ---")
ports_to_check = [80, 443, 554, 6123, 8080, 8554, 8899, 3478, 4747, 5000,
                  6000, 6124, 6125, 7000, 7070, 8000, 8888, 9000, 9090,
                  10000, 10554, 34567, 37777, 49152]

open_ports = []
for port in ports_to_check:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex((target, port))
        if result == 0:
            open_ports.append(port)
            print(f"  Port {port}: OPEN")
        s.close()
    except Exception:
        pass

if not open_ports:
    print("  No open TCP ports found in common range.")

# Also try UDP on port 6123 (camera might only use UDP)
print(f"\n--- Testing UDP on {target}:6123 ---")
try:
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.settimeout(2)
    udp_sock.sendto(b'\x00' * 4, (target, 6123))
    try:
        data, addr = udp_sock.recvfrom(1024)
        print(f"  UDP 6123: Got response! ({len(data)} bytes from {addr})")
    except socket.timeout:
        print(f"  UDP 6123: No response (timeout)")
    udp_sock.close()
except Exception as e:
    print(f"  UDP 6123: Error - {e}")

# Scan other IPs on the subnet
if local_ip:
    subnet = ".".join(local_ip.split(".")[:3])
    print(f"\n--- Scanning subnet {subnet}.0/24 for any device ---")
    found = []
    for i in range(1, 20):
        ip = f"{subnet}.{i}"
        if ip == local_ip:
            continue
        r = run(f"ping -c 1 -W 1 {ip}")
        if "bytes from" in r:
            found.append(ip)
            print(f"  {ip}: ALIVE")
    if not found:
        print("  No other devices found nearby")

print("\n" + "=" * 50)
print("COPY EVERYTHING ABOVE AND PASTE TO CLAUDE")
print("=" * 50)
