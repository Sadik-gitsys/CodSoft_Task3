
sms = input("Enter SMS: ")

# Convert text into TF-IDF features
sms_vector = vectorizer.transform([sms])

# Predict
prediction = model.predict(sms_vector)

# Output
if prediction[0] == 1:
    print("Spam SMS")
else:
    print("Legitimate SMS")