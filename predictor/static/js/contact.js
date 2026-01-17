$(document).ready(function() {
        $('#contact-form').on('submit', function(e) {
            e.preventDefault();

            var $form = $(this);
            var $submitButton = $form.find('#cf-submit');
            var $loadingSpinner = $form.find('#contact-loading-spinner');
            var $messagesContainer = $('#contact-form-messages');

            // 1. Show loading spinner and disable button
            $loadingSpinner.show();
            $submitButton.prop('disabled', true);
            $messagesContainer.empty(); // Clear any previous messages

            // 2. Send AJAX request
            $.ajax({
                url: $form.attr('action'),
                type: $form.attr('method'),
                data: $form.serialize(),
                dataType: 'json',
                success: function(response) {
                    if (response.success) {
                        $messagesContainer.html('<div class="alert alert-success alert-dismissible fade show" role="alert">' + response.message + '<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button></div>');
                        $form[0].reset(); // Clear form fields on success
                    } else {
                        var errorHtml = '<div class="alert alert-danger alert-dismissible fade show" role="alert">Please correct the following errors:<br>';
                        $.each(response.errors, function(field, messages) {
                            // Format field name nicely for display
                            var formattedField = field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            errorHtml += '<strong>' + formattedField + ':</strong> ' + messages.join('<br>') + '<br>';
                        });
                        errorHtml += '<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button></div>';
                        $messagesContainer.html(errorHtml);
                    }
                },
                error: function(xhr, status, error) {
                    $messagesContainer.html('<div class="alert alert-danger alert-dismissible fade show" role="alert">An unexpected error occurred while sending your message. Please try again.<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button></div>');
                    console.error("AJAX Error:", status, error, xhr.responseText);
                },
                complete: function() {
                    // 3. Hide loading spinner and re-enable button
                    $loadingSpinner.hide();
                    $submitButton.prop('disabled', false);
                }
            });
});

        // This event listener handles closing the Bootstrap alerts
        // Ensure you have Bootstrap's JS loaded for data-dismiss="alert" to work.
        // If not, you can use the manual jQuery fadeOut method as discussed before.
        $(document).on('click', '.alert .close', function() {
            $(this).closest('.alert').alert('close'); // Requires Bootstrap JS
            // Or, if Bootstrap JS is an issue, use this:
            // $(this).closest('.alert').fadeOut(500, function() {
            //     $(this).remove();
            // });
        });
    });
