import face_recognition

# Load the known images
person1_image = face_recognition.load_image_file("person_1.jpg")
person2_image = face_recognition.load_image_file("person_2.jpg")
person3_image = face_recognition.load_image_file("person_3.jpg")

# Get the face encoding of each person. This can fail if no one is found in the photo.
person1_face_encoding = face_recognition.face_encodings(person1_image)[0] #using the first result because knowing there is only one person in the image
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]

# Create a list of all known face encodings
known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding,
    person3_face_encoding
]

# Load the image we want to check
unknown_face = face_recognition.load_image_file("unknown_3.jpg")

# Get face encodings for any people in the picture
unknown_face_encodings = face_recognition.face_encodings(unknown_face)

# There might be more than one person in the photo, so we need to loop over each face we found
for unknown_face_encoding in unknown_face_encodings:

    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding) #results is a list of Boolean values

    name = "Unknown"

    if results[0]:
        name = "Person 1"
    elif results[1]:
        name = "Person 2"
    elif results[2]:
        name = "Person 3"

    print(f"Found {name} in the photo!")
