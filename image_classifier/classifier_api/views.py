# image_classifier/classifier_api/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from .utils import predict_class
@api_view(['POST'])
@csrf_exempt
def classify_image(request):
    try:
        if 'image' not in request.FILES and 'image' not in request.data:
            return Response({'error': 'No image provided'}, status=400)
        
        if 'image' in request.FILES:
            # Handle file upload
            image = request.FILES['image'].read()
        else:
            # Handle base64 encoded image
            image_data = request.data['image']
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                # The image is already in the right format for processing
                image = image_data
            else:
                return Response({'error': 'Invalid image format'}, status=400)
        
        print("Image received successfully, starting prediction...")
        
        # Make prediction
        result = predict_class(image)
        
        print(f"Prediction successful: {result}")
        
        return Response({
            'result': result['class'],
            'confidence': result['confidence']
        })
    
    except Exception as e:
        import traceback
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return Response({'error': str(e)}, status=500)