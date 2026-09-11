# Person 2 - Face Detection & Face Recognition

This module provides face detection, face enrollment, face embeddings, and authorized/unknown face recognition for the anti-spoofing access-control project.

## Files
- `face_recognition/recognizer.py` - main Person 2 recognition module
- `face_recognition/__init__.py` - Python package initializer
- `requirements.txt` - Streamlit Cloud dependencies
- `.gitignore` - prevents local databases and secrets from being committed

## Recognition flow
Camera -> face detection -> pretrained InsightFace embedding -> cosine similarity -> authorized/unknown

## Integration contract
`FaceRecognizer.enroll(name, images)` returns `success`, `message`, `samples_saved`, and `samples_rejected`.

`FaceRecognizer.recognize(frame)` returns face detection status, face count, bounding box, recognition result, name, distance, and confidence.

Person 3 remains responsible for the final GRANTED/DENIED access-control decision.

## Streamlit Cloud
Install dependencies from `requirements.txt`. The first application run may download the InsightFace pretrained model. Do not commit the generated SQLite database or secrets to GitHub.
