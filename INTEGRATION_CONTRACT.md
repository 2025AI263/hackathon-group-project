# Person 2 Integration Contract

## Person 2 provides
- Face detection
- Face enrollment
- Pretrained face embeddings
- Face matching
- Authorized / Unknown result

## Enrollment
Call:

```python
result = recognizer.enroll(name, captured_images)
```

Expected result keys:
- `success`
- `message`
- `samples_saved`
- `samples_rejected`

## Recognition
Call:

```python
result = recognizer.recognize(frame)
```

Expected result keys include:
- `face_detected`
- `face_count`
- `bbox`
- `recognized`
- `name`
- `distance`
- `confidence`

## Person 3 decision
Person 2 must not grant access. Person 3 applies:
- SPOOF -> DENIED
- LIVE + UNKNOWN -> DENIED
- LIVE + AUTHORIZED -> GRANTED
