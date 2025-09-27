models used:
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- https://huggingface.co/TalTechNLP/whisper-large-v3-turbo-et-verbatim
- https://huggingface.co/datasets/TalTechNLP/grammar_et

to run with cpu only:
<code>docker compose up --build -d</code>

to run with fast as phuc gpu go brr:
<code>docker build -f Dockerfile.gpu -t meeting-whisperer-gpu</code>
<code>docker-compose -f docker-compose.gpu.yml up --build </code>
