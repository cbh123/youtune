# YouTune

<div style="position: relative; padding-bottom: 64.5933014354067%; height: 0;"><iframe src="https://www.loom.com/embed/193fa040b8074f44bb5ddabd4dd42b01?sid=a142d29d-e2e9-439f-8ebd-6485a072b8e2" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

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
