def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Important: Match training preprocessing
    img = img.astype('float32') / 255.0  # If you didn't use MobileNet preprocessing
    
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)[0]
    print("Raw predictions:", predictions)  # Debug output
    
    if np.isnan(predictions).any():
        raise ValueError("Model returned NaN predictions - check training data")
    
    return {
        'class': CLASS_NAMES[np.argmax(predictions)],
        'confidence': float(np.max(predictions))
    }