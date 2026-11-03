# AI Voice Comparison Lab

Independent CPU-based voice cloning benchmarks for two reference voices, using the same movie narration script. Each model has its own GitHub Actions workflow and uploads artifacts independently. Outputs are experimental; inspect Actions logs for failures.

Reference recordings are fetched from the existing Top10-Movies repository at run time. Only use voices with appropriate consent and rights.

Models: Chatterbox Turbo, Chatterbox TTS, Chatterbox Multilingual, F5-TTS, CosyVoice, Fish Speech. The last three may be slow or fail under GitHub-hosted CPU and disk limits.

Go to Actions to see independent jobs and download WAV files. No paid APIs are required, but GitHub Actions usage limits apply.
