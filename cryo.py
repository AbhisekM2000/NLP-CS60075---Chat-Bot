import jsonlines


with jsonlines.open('your-filename.jsonl') as f:
    for line in f.iter():
        print line['doi'] # or whatever else you'd like to do