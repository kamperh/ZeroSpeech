#!/usr/bin/env python

"""
Synthesize the codes from a text file giving repeated code indices.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
import argparse
import json
import librosa
import numpy as np
import pyloudnorm
import sys
import torch

from model import Encoder, Decoder
from preprocess import preemphasis


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "checkpoint", type=str, help="model checkpoint"
        )
    parser.add_argument(
        "code_indices_fn", type=str, help="text file with code indices"
        )
    parser.add_argument(
        "--speaker", type=str, default="V001",
        help="speaker identifier (default: %(default)s)"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Code indices
    code_indices_fn = Path(args.code_indices_fn)
    print("Reading: {}".format(code_indices_fn))
    code_indices = np.loadtxt(code_indices_fn, dtype=np.int)

    # Speakers
    with open(Path("datasets/2019/english/speakers.json")) as f:
        speakers = sorted(json.load(f))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(
        in_channels=80, channels=768, n_embeddings=512, embedding_dim=64,
        jitter=0.5
        )
    decoder = Decoder(
        in_channels=64, conditioning_channels=128, n_speakers=102,
        speaker_embedding_dim=64, mu_embedding_dim=256, rnn_channels=896,
        fc_channels=256, bits=8, hop_length=160,
        )
    decoder.to(device)

    print("Reading: {}".format(args.checkpoint))
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage
        )
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    # Codes
    embedding = encoder.codebook.embedding.cpu().numpy()
    codes = np.array([embedding[code_indices]])

    # Synthesize
    z = torch.FloatTensor(codes).to(device)
    speaker = torch.LongTensor([speakers.index(args.speaker)]).to(device)
    with torch.no_grad():
        output = decoder.generate(z, speaker)

    wav_fn = Path(code_indices_fn.stem).with_suffix(".wav")
    print("Writing: {}".format(wav_fn))
    librosa.output.write_wav(wav_fn, output.astype(np.float32), sr=16000)

    # # Loadness
    # meter = pyloudnorm.Meter(16000)
    # output_loudness = meter.integrated_loudness(output)
    # output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)



if __name__ == "__main__":
    main()
