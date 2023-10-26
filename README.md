# YouTune

[YouTune Video Walkthrough](https://www.loom.com/share/193fa040b8074f44bb5ddabd4dd42b01?sid=4b09aa1b-5cd6-4e4f-a538-d3d62cb1bdc0)

YouTune makes it really easy to fine-tune SDXL on images from YouTube videos. Just give it a URL and a model name on Replicate, and it'll download the video, take screenshots of every 50 frames, remove near duplicates and very light/dark images, and create a training for you.

## Setup

Clone this repo, and setup and activate a virtualenv:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
```

Then, install the dependencies:
`pip install -r requirements.txt`

Make a [Replicate](https://replicate) account and set your token:

`export REPLICATE_API_TOKEN=<token>`

## Run it!

```bash
python tune.py <youtube-url>
```
