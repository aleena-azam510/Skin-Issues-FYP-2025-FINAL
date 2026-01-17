from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from .models import MyAIReport, Notification


@receiver(post_save, sender=MyAIReport)
def notify_doctor_on_report_sent(sender, instance, created, **kwargs):
    """
    Notify doctor when a new AI report is sent to them
    """
    if created and instance.doctor:
        Notification.objects.create(
            recipient=instance.doctor,
            message=(
                f"New AI report received from "
                f"{instance.user.get_full_name() or instance.user.username}."
            ),
            notification_type="report_sent",
            related_report=instance
        )


@receiver(pre_save, sender=MyAIReport)
def notify_user_on_report_reviewed(sender, instance, **kwargs):
    """
    Notify user only when report status changes to 'reviewed'
    """
    if not instance.pk:
        return  # new object, skip

    try:
        previous = MyAIReport.objects.get(pk=instance.pk)
    except MyAIReport.DoesNotExist:
        return

    # Send notification ONLY when status changes to reviewed
    if previous.status != "reviewed" and instance.status == "reviewed":
        if instance.doctor:
            doctor_name = instance.doctor.get_full_name() or instance.doctor.username
        else:
            doctor_name = "your doctor"

        Notification.objects.create(
            recipient=instance.user,
            message=f"Your AI report has been reviewed by Dr. {doctor_name}.",
            notification_type="report_reviewed",
            related_report=instance
        )
