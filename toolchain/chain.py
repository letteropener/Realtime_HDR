import subprocess


if __name__ == "__main__":
    subprocess.run(["python", "comparagram_RGB_v1.py", "&&", "python","comparasum_RGB.py","&&","python","hdr_CCRF_RGB.py"], stdout=subprocess.PIPE)