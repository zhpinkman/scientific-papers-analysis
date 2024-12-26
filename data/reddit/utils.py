import argparse
from IPython import embed
import json
from collections import defaultdict
from datetime import datetime, timezone
import pandas as pd
import re


def utc_to_datetime(utc_timestamp):
    return datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)


def extract_plain_text(markdown_text):
    # Remove markdown links (e.g., [text](url))
    plain_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown_text)
    # Remove bold and italic markdown formatting (*text* or **text**)
    plain_text = re.sub(r"\*{1,2}([^\*]+)\*{1,2}", r"\1", plain_text)
    # Remove markdown headings or dividers (e.g., "---", "### Text")
    plain_text = re.sub(r"^\s*[-#]+.*$", "", plain_text, flags=re.MULTILINE)
    # Replace escaped newlines with actual newlines
    plain_text = re.sub(r"\\n", "\n", plain_text)
    # Strip leading/trailing whitespace
    # remove all the extra white spaces
    plain_text = re.sub(r"\s+", " ", plain_text)
    return plain_text.strip()


def process_writing_prompts():
    with open("r_WritingPrompts_posts.jsonl", "r") as f:
        posts_data = [json.loads(line) for line in f.readlines()]
        f.close()
    with open("r_WritingPrompts_comments.jsonl", "r") as f:
        comments_data = [json.loads(line) for line in f.readlines()]
        f.close()

    posts_id_key = "name"
    posts_text_key = "selftext"
    posts_title_key = "title"
    comments_parent_id_key = "parent_id"
    comments_id_key = "name"
    comments_text_key = "body"
    time_key = "created_utc"

    prefix_to_filter_on = "[WP]"

    # convert utc timestamps to datetime objects

    posts_data = [
        {
            key: value
            for key, value in data.items()
            if key in [posts_id_key, posts_text_key, posts_title_key, time_key]
        }
        for data in posts_data
    ]
    posts_data = [
        data
        for data in posts_data
        if data[posts_title_key].startswith(prefix_to_filter_on)
    ]
    posts_data = {data[posts_id_key]: data for data in posts_data}

    comments_data = [
        {
            key: value
            for key, value in data.items()
            if key
            in [comments_parent_id_key, comments_text_key, time_key, comments_id_key]
        }
        for data in comments_data
    ]
    comments_data = [
        data for data in comments_data if data[comments_parent_id_key] in posts_data
    ]
    comments_data = {data[comments_id_key]: data for data in comments_data}
    # change the text in comments_text_key property to the modified version using extract_plain_text
    for key in comments_data.keys():
        comments_data[key][comments_text_key] = extract_plain_text(
            comments_data[key][comments_text_key]
        )
    # filter the comments to only those comments that are longer than 100 words
    comments_data = {
        comment_id: comment
        for comment_id, comment in comments_data.items()
        if len(comment[comments_text_key].split()) > 200
    }

    # filter the data and exclude the ones that contain the phrase 'this submission has been removed'
    comments_data = {
        comment_id: comment
        for comment_id, comment in comments_data.items()
        if "this submission has been removed" not in comment[comments_text_key].lower()
        and "welcome to the prompt" not in comment[comments_text_key].lower()
    }

    # posts_to_comments = defaultdict(list)
    # for comment_id, comment in comments_data.items():
    #     posts_to_comments[comment[comments_parent_id_key]].append(comment_id)

    # convert the datetime utc objects to datetime objects
    for key in posts_data.keys():
        posts_data[key][time_key] = utc_to_datetime(posts_data[key][time_key])

    for key in comments_data.keys():
        comments_data[key][time_key] = utc_to_datetime(comments_data[key][time_key])

    # show the number of comments grouped by month and year
    comments_df = pd.DataFrame(comments_data.values())
    comments_df["month"] = comments_df[time_key].apply(lambda x: x.month)
    comments_df["year"] = comments_df[time_key].apply(lambda x: x.year)
    comments_df.to_csv("filtered_comments.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process writing prompts")
    parser.add_argument("--process", required=True, type=str)

    args = parser.parse_args()

    if args.process == "writing_prompts":
        process_writing_prompts()
