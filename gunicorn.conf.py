# Gunicorn reads this file automatically when it starts (it must sit next to
# app.py). Render's start command "gunicorn app:app" picks it up as is.

# One worker process: the rate limits are counted in that process's memory,
# so more processes would each keep their own count.
workers = 1

# Two threads, so a second visitor isn't stuck waiting while someone's
# 10-page Premium upload is being read. Kept low for the 512 MB free server.
threads = 2

# Reading several pages together can take a few minutes. Gunicorn's default
# (30 seconds) would cut those requests off halfway.
timeout = 300
