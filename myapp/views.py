from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .data_loading import final_model  # replace this with the function or class that runs your model

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = request.POST  # you might need to do some preprocessing on this data
        prediction = final_model(data)  # replace this with how you would call your model function or method
        return JsonResponse({'prediction': prediction})
