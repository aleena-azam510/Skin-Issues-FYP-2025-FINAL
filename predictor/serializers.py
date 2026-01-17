from rest_framework import serializers
from .models import SkinCondition, Remedy

class RemedySerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()
    directions = serializers.CharField(source='formatted_directions', read_only=True)

    class Meta:
        model = Remedy
        fields = ['title', 'directions', 'amount', 'image_url']

    def get_image_url(self, obj):
        if obj.image:
            return obj.image.url
        return None

class SkinConditionSerializer(serializers.ModelSerializer):
    remedies = RemedySerializer(many=True, source='remedy_set', read_only=True)
    causes = serializers.ListField(source='causes_list', read_only=True)
    symptoms = serializers.ListField(source='symptoms_list', read_only=True)

    class Meta:
        model = SkinCondition
        fields = ['causes', 'symptoms', 'remedies']
