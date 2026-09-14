"""OPTIONAL PCAP feature extraction (Scapy).

This is an explicitly optional pipeline. It is NEVER imported by the core
flow-based build path, so a missing Scapy install or absent PCAPs can never
block dataset processing.

It extracts ONLY defensibly-computable packet features into PacketFeatureRecord:
    - packet_length  (frame length)
    - ttl            (IP TTL)
    - tcp_flags      (TCP flag string, if TCP)
    - payload_bytes  (transport payload length)
    - timestamp      (capture time)
Derived per-flow-ish signals a caller may compute from a stream:
    - IAT            (inter-arrival time between consecutive packets)
    - retransmission indicator (heuristic: duplicate TCP seq on a 5-tuple)

It does NOT fabricate payload content, and does not synthesise any feature that
cannot be read from the packet. If Scapy is unavailable, ``pcap_available()``
returns False and ``iter_packet_features`` raises a clear ImportError.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from ..data.schema import PacketFeatureRecord


def pcap_available() -> bool:
    """True if Scapy is importable (optional dependency)."""
    try:
        import scapy.all  # noqa: F401
        return True
    except Exception:
        return False


def iter_packet_features(
    pcap_path: str,
    *,
    dataset: str,
    max_packets: Optional[int] = None,
) -> Iterator[PacketFeatureRecord]:
    """Stream PacketFeatureRecords from a PCAP using Scapy's PcapReader.

    Raises ImportError if Scapy is not installed (optional dependency).
    Uses PcapReader (streaming) — does not load the whole capture into RAM.
    """
    if not pcap_available():
        raise ImportError(
            "Scapy is not installed. PCAP extraction is optional; install with "
            "`pip install scapy` to enable it. The core flow pipeline does not "
            "require it."
        )
    from scapy.all import PcapReader  # type: ignore
    from scapy.layers.inet import IP, TCP, UDP  # type: ignore

    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP not found: {path}")

    count = 0
    with PcapReader(str(path)) as reader:
        for pkt in reader:
            ttl = None
            payload_bytes = None
            tcp_flags = None
            packet_length = len(pkt) if pkt is not None else None
            ts = None
            try:
                ts = datetime.fromtimestamp(float(pkt.time), tz=timezone.utc)
            except Exception:
                ts = None

            if pkt.haslayer(IP):
                ip = pkt[IP]
                ttl = int(ip.ttl)
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                tcp_flags = str(tcp.flags)
                try:
                    payload_bytes = len(bytes(tcp.payload))
                except Exception:
                    payload_bytes = None
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                try:
                    payload_bytes = len(bytes(udp.payload))
                except Exception:
                    payload_bytes = None

            yield PacketFeatureRecord(
                dataset=dataset,
                flow_source_index=-1,  # not linked to a specific flow row here
                packet_index=count,
                timestamp=ts,
                packet_length=packet_length,
                ttl=ttl,
                tcp_flags=tcp_flags,
                payload_bytes=payload_bytes,
                source="pcap",
            )
            count += 1
            if max_packets is not None and count >= max_packets:
                return
