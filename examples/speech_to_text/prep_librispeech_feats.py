#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import pandas as pd
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    get_n_speech_features
)
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    extract_speech_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)

from tqdm import tqdm


log = logging.getLogger(__name__)

SPLITS = [
    "train-clean-100",
    "dev-clean",
    "dev-other",
  #  "test-clean",
    "test-other",
    "train-clean-360",
    "train-other-500",
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args, data_cfg, n_speech_features):
    out_root = Path(args.output_root).absolute()
    feature_root = out_root / args.folder_name
    feature_root.mkdir(exist_ok=True)
    os.system(f"cp {args.config_yaml} {feature_root}")
    for split in SPLITS:
        print(f"Fetching split {split}...")
        df = pd.read_csv(os.path.join(out_root, f"{split}_wav.tsv"), sep='\t')
        print("Extracting log mel filter bank features...")
        for index, row in tqdm(df.iterrows()):
            wav_path = row['audio']
            sample_id = row['id']
            tgt_text = row['tgt_text']
            speaker = row['speaker']
            extract_speech_features(feature_root, data_cfg, n_speech_features, wav_path, sample_id, tgt_text, speaker)
    # # Pack features into ZIP
    # zip_path = out_root / "fbank80.zip"
    # print("ZIPing features...")
    # create_zip(feature_root, zip_path)
    # print("Fetching ZIP manifest...")
    # zip_manifest = get_zip_manifest(zip_path)
    # # Generate TSV manifest
    # print("Generating manifest...")
    # train_text = []
    # for split in SPLITS:
    #     manifest = {c: [] for c in MANIFEST_COLUMNS}
    #     dataset = LIBRISPEECH(out_root.as_posix(), url=split)
    #     for wav, sample_rate, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
    #         sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
    #         manifest["id"].append(sample_id)
    #         manifest["audio"].append(zip_manifest[sample_id])
    #         duration_ms = int(wav.size(1) / sample_rate * 1000)
    #         manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
    #         manifest["tgt_text"].append(utt)
    #         manifest["speaker"].append(spk_id)
    #     save_df_to_tsv(
    #         pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
    #     )
    #     if split.startswith("train"):
    #         train_text.extend(manifest["tgt_text"])
    # # Generate vocab
    # vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    # spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    # with NamedTemporaryFile(mode="w") as f:
    #     for t in train_text:
    #         f.write(t + "\n")
    #     gen_vocab(
    #         Path(f.name),
    #         out_root / spm_filename_prefix,
    #         args.vocab_type,
    #         args.vocab_size,
    #     )
    # # Generate config YAML
    # gen_config_yaml(
    #     out_root, spm_filename_prefix + ".model", specaugment_policy="ld"
    # )
    # # Clean up
    # shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", default="/disks/md1-8T/users/b.gcr/datasets/eng/librispeech/fairseq/", type=str)
    parser.add_argument("--folder_name", default="features_20210309", type=str)
    parser.add_argument("--config-yaml", default="/disks/md1-8T/users/b.gcr/datasets/eng/librispeech/fairseq/config_40fb_precomputation.yaml", type=str)
    args = parser.parse_args()

    data_cfg = S2TDataConfig(args.config_yaml)
    n_speech_features = get_n_speech_features(data_cfg)
    process(args, data_cfg, n_speech_features)

if __name__ == "__main__":
    main()
