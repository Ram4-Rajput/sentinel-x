"""Shared fixtures: tiny synthetic rows that mirror each REAL dataset schema.

These are hand-written literals (per writing-good-tests: derive expectations by
hand, never from the code under test). No real data files are read.
"""

import sys
from pathlib import Path

import pytest

# Make the src/ package importable without installation.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def cic_rows():
    """Two CIC-IDS2018 rows (subset of the 80 CICFlowMeter columns)."""
    return [
        {
            "Dst Port": "0", "Protocol": "0", "Timestamp": "14/02/2018 08:31:01",
            "Flow Duration": "112641719",  # microseconds
            "Tot Fwd Pkts": "3", "Tot Bwd Pkts": "0",
            "TotLen Fwd Pkts": "0", "TotLen Bwd Pkts": "0",
            "Flow Byts/s": "0", "Flow Pkts/s": "0.0266",
            "Flow IAT Mean": "56320859.5", "Fwd IAT Mean": "56320859.5",
            "SYN Flag Cnt": "0", "ACK Flag Cnt": "0",
            "FIN Flag Cnt": "0", "RST Flag Cnt": "0",
            "Init Fwd Win Byts": "-1", "Fwd Header Len": "0", "Bwd Header Len": "0",
            "Label": "Benign",
        },
        {
            "Dst Port": "22", "Protocol": "6", "Timestamp": "14/02/2018 08:33:50",
            "Flow Duration": "2000000",
            "Tot Fwd Pkts": "5", "Tot Bwd Pkts": "2",
            "TotLen Fwd Pkts": "500", "TotLen Bwd Pkts": "120",
            "Flow Byts/s": "Infinity", "Flow Pkts/s": "NaN",  # known Inf/NaN issue
            "Flow IAT Mean": "1000", "Fwd IAT Mean": "900",
            "SYN Flag Cnt": "1", "ACK Flag Cnt": "1",
            "FIN Flag Cnt": "0", "RST Flag Cnt": "0",
            "Init Fwd Win Byts": "8192", "Fwd Header Len": "40", "Bwd Header Len": "20",
            "Label": "SSH-Bruteforce",
        },
    ]


@pytest.fixture
def ciciot_rows():
    """Two CICIoT2023 rows (subset of the 47 columns; NO timestamp/IP)."""
    return [
        {
            "flow_duration": "0.0", "Header_Length": "757", "Protocol Type": "6",
            "Rate": "23.67", "Srate": "23.67",
            "syn_count": "0", "ack_count": "0", "fin_count": "0", "rst_count": "0",
            "Tot sum": "9627", "IAT": "83340575.0", "Number": "9.5",
            "label": "BenignTraffic",
        },
        {
            "flow_duration": "0.0339", "Header_Length": "54", "Protocol Type": "6",
            "Rate": "1.19", "Srate": "1.19",
            "syn_count": "1", "ack_count": "0", "fin_count": "0", "rst_count": "0",
            "Tot sum": "572", "IAT": "83330863.0", "Number": "9.5",
            "label": "DDoS-SYN_Flood",
        },
    ]


@pytest.fixture
def ctu_rows():
    """CTU-13 .binetflow rows (15 columns) with entities + time + free-text label."""
    return [
        {
            "StartTime": "2011/08/10 09:46:59.607825", "Dur": "1.026539",
            "Proto": "tcp", "SrcAddr": "147.32.84.59", "Sport": "1577",
            "Dir": "->", "DstAddr": "147.32.84.229", "Dport": "6881",
            "State": "S_RA", "sTos": "0", "dTos": "0",
            "TotPkts": "4", "TotBytes": "276", "SrcBytes": "156",
            "Label": "flow=Background-Established-cmpgw-CVUT",
        },
        {
            "StartTime": "2011/08/10 09:47:10.100000", "Dur": "0.5",
            "Proto": "tcp", "SrcAddr": "147.32.84.165", "Sport": "1040",
            "Dir": "->", "DstAddr": "147.32.84.59", "Dport": "80",
            "State": "CON", "sTos": "0", "dTos": "0",
            "TotPkts": "10", "TotBytes": "1000", "SrcBytes": "400",
            "Label": "flow=From-Botnet-V42-TCP-Attempt",
        },
        {
            "StartTime": "2011/08/10 09:47:20.000000", "Dur": "0",  # zero dur -> rate None
            "Proto": "udp", "SrcAddr": "147.32.84.59", "Sport": "53",
            "Dir": "->", "DstAddr": "8.8.8.8", "Dport": "53",
            "State": "CON", "sTos": "0", "dTos": "0",
            "TotPkts": "2", "TotBytes": "200", "SrcBytes": "100",
            "Label": "flow=Normal-DNS",
        },
    ]


@pytest.fixture
def unsw_raw_values():
    """One header-less UNSW-NB15 raw row as a 49-value sequence (order per
    RAW_COLUMN_ORDER)."""
    # srcip,sport,dstip,dsport,proto,state,dur,sbytes,dbytes,sttl,dttl,sloss,dloss,
    # service,Sload,Dload,Spkts,Dpkts,swin,dwin,stcpb,dtcpb,smeansz,dmeansz,
    # trans_depth,res_bdy_len,Sjit,Djit,Stime,Ltime,Sintpkt,Dintpkt,tcprtt,synack,
    # ackdat,is_sm_ips_ports,ct_state_ttl,ct_flw_http_mthd,is_ftp_login,ct_ftp_cmd,
    # ct_srv_src,ct_srv_dst,ct_dst_ltm,ct_src_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,
    # ct_dst_src_ltm,attack_cat,Label
    return [
        "59.166.0.0", "1390", "149.171.126.6", "53", "udp", "CON", "0.001055",
        "132", "164", "31", "29", "0", "0", "dns", "500473.9", "621800.9",
        "2", "2", "255", "255", "0", "0", "66", "82", "0", "0", "0.017", "0.013",
        "1421927414", "1421927414", "0.5", "0.4", "0", "0", "0", "0", "3", "0",
        "0", "0", "3", "7", "1", "3", "1", "1", "1", "", "0",
    ]


@pytest.fixture
def unsw_partition_rows():
    """Two rows from the labeled partition CSV (headered; lowercase; no srcip)."""
    return [
        {
            "id": "1", "dur": "0.000011", "proto": "udp", "service": "-",
            "state": "INT", "spkts": "2", "dpkts": "0", "sbytes": "496",
            "dbytes": "0", "rate": "90909", "sttl": "254", "dttl": "0",
            "swin": "0", "dwin": "0", "Sload": "180363632",
            "attack_cat": "Normal", "Label": "0",
        },
        {
            "id": "2", "dur": "0.121", "proto": "tcp", "service": "-",
            "state": "FIN", "spkts": "6", "dpkts": "4", "sbytes": "258",
            "dbytes": "172", "rate": "74", "sttl": "252", "dttl": "254",
            "swin": "255", "dwin": "255", "Sload": "14158",
            "attack_cat": "Exploits", "Label": "1",
        },
    ]
