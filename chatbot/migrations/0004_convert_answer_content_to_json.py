# chatbot/migrations/0004_convert_old_answer_content.py  <-- IMPORTANT: USE YOUR ACTUAL FILENAME HERE

import json
from django.db import migrations

def convert_old_answers_to_json(apps, schema_editor):
    """
    Converts existing Answer.content from plain strings to JSONField format.
    Wraps old string content into a [{"type": "paragraph", "text": "..."}] structure.
    """
    Answer = apps.get_model('chatbot', 'Answer') # Get the Answer model
    db_alias = schema_editor.connection.alias # Get the database alias

    for answer in Answer.objects.using(db_alias).all():
        # Check if the content is currently a string (this indicates old, unconverted data)
        # It's important to check `isinstance(answer.content, str)` because if the migration
        # fails halfway or runs again, you don't want to re-wrap already valid JSON.
        if isinstance(answer.content, str):
            # Wrap the old plain string into the new JSON structure
            # Example: "dry skin is a type of skin" becomes [{"type": "paragraph", "text": "dry skin is a type of skin"}]
            answer.content = [{"type": "paragraph", "text": answer.content}]
            answer.save()
        elif answer.content is None:
            # Handle cases where content might be None (if your field was temporarily null=True)
            # Initialize it as an empty list (empty JSON array)
            answer.content = []
            answer.save()
        # If it's already a list or dict, it's presumed to be valid JSON, so we do nothing.

def reverse_convert_old_answers_to_json(apps, schema_editor):
    """
    Reverts the Answer.content from JSONField format back to a plain string.
    (Optional, but good for reversibility. May lose data if JSON was complex.)
    """
    Answer = apps.get_model('chatbot', 'Answer')
    db_alias = schema_editor.connection.alias
    for answer in Answer.objects.using(db_alias).all():
        if isinstance(answer.content, list) and answer.content:
            # Try to extract text from the first paragraph-like block
            first_block_text = ""
            if isinstance(answer.content[0], dict) and 'text' in answer.content[0]:
                first_block_text = answer.content[0]['text']
            answer.content = first_block_text # Revert to plain string
            answer.save()
        elif answer.content is None:
            answer.content = "" # Or whatever default string you want for nulls
            answer.save()


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0003_alter_answer_content'), # Make sure this points to the migration that changed to JSONField
        # If you had another migration after 0003 (e.g., to make content nullable),
        # add it here too. For example: ('chatbot', '0004_alter_answer_content_nullable'),
    ]

    operations = [
        # This is where you add the operation to run your Python function
        migrations.RunPython(convert_old_answers_to_json, reverse_convert_old_answers_to_json),
    ]