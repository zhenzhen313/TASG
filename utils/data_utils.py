import os
import pandas as pd
import emoji
from utils.tokenizer_new import preprocess_data

def load_data(args, replace_emoji=True, senti_on = True):
    # data loading
    train_tsv = os.path.join(args.data_dir, args.dataset, "train.tsv")
    dev_tsv = os.path.join(args.data_dir, args.dataset, "dev.tsv")
    test_tsv = os.path.join(args.data_dir, args.dataset, "test.tsv")

    test_df = pd.read_csv(test_tsv, sep="\t")
    train_df = pd.read_csv(train_tsv, sep="\t")
    val_df = pd.read_csv(dev_tsv, sep="\t")

    test_df = test_df.rename(
        {
            "#1 Label": "sentiment",
            "#2 ImageID": "image_id",
            "#3 String": "tweet_content",
            "#4 String": "target",
        },
        axis=1,
    )
    train_df = train_df.rename(
        {
            "#1 Label": "sentiment",
            "#2 ImageID": "image_id",
            "#3 String": "tweet_content",
            "#4 String": "target",
        },
        axis=1,
    ).drop(["index"], axis=1) 
    val_df = val_df.rename(
        {
            "#1 Label": "sentiment",
            "#2 ImageID": "image_id",
            "#3 String": "tweet_content",
            "#4 String": "target",
        },
        axis=1,
    ).drop(["index"], axis=1)
    
    if replace_emoji:
        train_df['tweet_content'] = train_df['tweet_content'].apply(emoji.replace_emoji)
        val_df['tweet_content'] = val_df['tweet_content'].apply(emoji.replace_emoji)
        test_df['tweet_content'] = test_df['tweet_content'].apply(emoji.replace_emoji)

    if senti_on:
        train_senti_output = preprocess_data(train_tsv)
        val_senti_output = preprocess_data(dev_tsv)
        test_senti_output = preprocess_data(test_tsv)
    else:
         train_senti_output =None
         val_senti_output = None
         test_senti_output = None
        
    return train_df, val_df, test_df, train_senti_output, val_senti_output, test_senti_output