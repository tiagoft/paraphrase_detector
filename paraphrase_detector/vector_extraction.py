import paraphrase_detector
import pandas as pd
from tqdm import tqdm 
import os
from pathlib import Path
import numpy as np

def extract_from_json(json_file : str,
                      output_dir : str,
                      content : str,
                      key : str = None,
    ):
    output_dir = Path(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.read_json(json_file, lines=True)
    model = paraphrase_detector.get_model()
    for i in tqdm(range(len(df))):
        if key is None:
            this_key = f"{i:06d}"
        else:
            this_key = df.loc[i, key]
        full_fname = output_dir / f"{this_key}.npy"
        if os.path.exists(full_fname):
            continue
        text = df.loc[i, content]
        sentences = paraphrase_detector.get_sentences_from_text(text)
        embeddings = paraphrase_detector.get_embeddings_from_sentences(model, sentences)
        with open(full_fname, 'wb') as f:
            np.save(f, embeddings)
    return