from django.db import models
from django.db.models import JSONField # Import JSONField

class Answer(models.Model):
    content = models.JSONField(default=list) # Should now be back to this or whatever your final requirement is

    def __str__(self):
        if isinstance(self.content, list) and self.content:
            # Iterate through blocks to find the first displayable text
            for block in self.content:
                if block.get('type') in ['heading', 'paragraph', 'listItem', 'tip', 'sectionHeading']:
                    text_to_display = block.get('text', '').strip()
                    if text_to_display: # Only return if text is not empty after stripping
                        return text_to_display[:75] + '...' if len(text_to_display) > 75 else text_to_display
            return "Structured Answer Content (empty text blocks)" # Fallback if list has blocks but no text
        return "Empty Answer Content" # This will show if content is [], None, or not a list
    
class Question(models.Model):
    text = models.CharField(max_length=255)  # The exact question wording
    answer = models.ForeignKey(Answer, on_delete=models.CASCADE)
    keywords = models.TextField(blank=True, null=True)  # Optional comma-separated keywords/tags for filtering/search

    def __str__(self):
        return self.text