#!/usr/bin/env python3
"""
Camera finder script for StangWatch.

Run this while connected to the camera's AP mode hotspot.
It will:
1. Find the camera's IP (usually the gateway)
2. Scan for open ports
3. Try common RTSP URLs
4. Try to grab a frame with OpenCV

Usage:
    python scripts/find_camera.py

Then copy-paste the output back to Claude.
"""

import subprocess
import socket
import sys
import os

# Add backend to path so we can use OpenCV if available
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def get_gateway():
    """Get the default gateway IP (likely the camera in AP mode)."""
    try:
        result = subprocess.run(
            ["route", "-n", "get", "default"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "gateway:" in line:
                return line.split("gateway:")[-1].strip()
    except Exception:
        pass
    return None


def get_local_ip():
    """Get this machine's IP on the current network."""
    try:
        result = subprocess.run(
            ["ipconfig", "getifaddr", "en0"],
            capture_output=True, text=True, timeout=5,
        )
        ip = result.stdout.strip()
        if ip:
            return ip
    except Exception:
        pass
    return None


def scan_ports(ip, ports):
    """Check which ports are open on the given IP."""
    open_ports = []
    for port in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((ip, port))
            if result == 0:
                open_ports.append(port)
            sock.close()
        except Exception:
            pass
    return open_ports


def try_rtsp(ip, port, paths):
    """Try RTSP URLs and see if OpenCV can connect."""
    try:
        import cv2
    except ImportError:
        print("  OpenCV not available, skipping RTSP test")
        return []

    working = []
    for path in paths:
        url = f"rtsp://{ip}:{port}{path}"
        print(f"  Trying {url} ...", end=" ", flush=True)
        cap = cv2.VideoCapture(url)
        # Give it a moment
        import time
        time.sleep(2)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"SUCCESS! Frame: {frame.shape}")
            working.append(url)
        else:
            print("no")
    return working


def try_http_stream(ip, port, paths):
    """Try HTTP stream URLs."""
    try:
        import cv2
    except ImportError:
        return []

    working = []
    for path in paths:
        url = f"http://{ip}:{port}{path}"
        print(f"  Trying {url} ...", end=" ", flush=True)
        cap = cv2.VideoCapture(url)
        import time
        time.sleep(2)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"SUCCESS! Frame: {frame.shape}")
            working.append(url)
        else:
            print("no")
    return working


def main():
    print("=" * 60)
    print("StangWatch Camera Finder")
    print("=" * 60)

    local_ip = get_local_ip()
    gateway = get_gateway()
    print(f"\nYour Mac IP: {local_ip}")
    print(f"Gateway IP:  {gateway}")

    if not gateway:
        print("\nCould not find gateway. Are you connected to the camera's WiFi?")
        return

    # Determine subnet to scan
    subnet = ".".join(gateway.split(".")[:3])
    print(f"Subnet:      {subnet}.0/24")

    # Scan common ports on the gateway (most likely the camera)
    print(f"\n--- Scanning ports on {gateway} ---")
    common_ports = [
        80, 443, 554, 1554, 5000, 8000, 8080, 8081, 8088, 8443, 8554,
        8899, 9000, 9090, 10554, 34567, 37777, 49152, 49153, 49154,
        6666, 7070, 3478, 4747, 5050, 8888,
    ]
    open_ports = scan_ports(gateway, common_ports)
    if open_ports:
        print(f"Open ports: {open_ports}")
    else:
        print("No common ports open on gateway.")

    # Also scan a few nearby IPs in case camera isn't the gateway
    print(f"\n--- Scanning nearby IPs ---")
    for last_octet in range(1, 10):
        ip = f"{subnet}.{last_octet}"
        if ip == gateway or ip == local_ip:
            continue
        ports = scan_ports(ip, [80, 554, 8080, 8554])
        if ports:
            print(f"  {ip}: open ports {ports}")

    # Try RTSP on any open 554/8554 ports
    all_working = []
    rtsp_paths = [
        "/", "/live", "/live/ch0", "/live/ch00_0",
        "/stream1", "/stream0", "/cam/realmonitor",
        "/h264", "/h264_stream", "/video.h264",
        "/ch0_0.h264", "/0", "/1", "/11", "/12",
        "/Streaming/Channels/1", "/Streaming/Channels/101",
        "/user=admin&password=&channel=1&stream=0.sdp",
        "/onvif1", "/media/video1",
    ]

    for ip in [gateway] + [f"{subnet}.{i}" for i in range(1, 10) if f"{subnet}.{i}" not in (gateway, local_ip)]:
        ports = scan_ports(ip, [554, 8554, 10554])
        for port in ports:
            print(f"\n--- Testing RTSP on {ip}:{port} ---")
            working = try_rtsp(ip, port, rtsp_paths)
            all_working.extend(working)

    # Try HTTP streams on any open 80/8080 ports
    http_paths = [
        "/video", "/video.cgi", "/videostream.cgi",
        "/mjpg/video.mjpg", "/cgi-bin/mjpg/video.cgi",
        "/stream", "/live",
    ]
    for ip in [gateway]:
        ports = scan_ports(ip, [80, 8080, 8081, 8088])
        for port in ports:
            print(f"\n--- Testing HTTP stream on {ip}:{port} ---")
            working = try_http_stream(ip, port, http_paths)
            all_working.extend(working)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if all_working:
        print("Working stream URLs:")
        for url in all_working:
            print(f"  {url}")
    else:
        print("No working streams found.")
        print("\nCopy-paste everything above and share it with Claude.")
        print("The open ports and IPs will help figure out the next step.")

    print(f"\nGateway: {gateway}")
    print(f"Open ports on gateway: {open_ports}")


if __name__ == "__main__":
    main()
