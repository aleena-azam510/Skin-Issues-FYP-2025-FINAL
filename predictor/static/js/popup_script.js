// popup_script.js

document.addEventListener('DOMContentLoaded', function() {
    const popupOverlay = document.getElementById('login-popup-overlay');
    const loginButton = document.getElementById('popup-login-button');
    const closeButton = document.getElementById('popup-close-button');

    // Function to show the popup
    function showPopup() {
        if (popupOverlay) {
            popupOverlay.classList.add('active');
            // Prevent scrolling on the background when popup is active
            document.body.style.overflow = 'hidden';
        }
    }

    // Function to hide the popup
    function hidePopup() {
        if (popupOverlay) {
            popupOverlay.classList.remove('active');
            document.body.style.overflow = ''; // Restore scrolling
        }
    }

    // Event listener for the login button in the popup
    if (loginButton) {
        loginButton.addEventListener('click', function() {
            // Redirect to your Django auth_page URL
            window.location.href = '/auth/'; // Use the hardcoded path or get from Django URL (more complex for JS)
        });
    }

    // Event listener for the close button in the popup
    if (closeButton) {
        closeButton.addEventListener('click', hidePopup);
    }

    // Optional: Hide popup if user clicks outside the content
    if (popupOverlay) {
        popupOverlay.addEventListener('click', function(event) {
            if (event.target === popupOverlay) {
                hidePopup();
            }
        });
    }

    // --- How to trigger the popup (choose one method) ---

    // Method 1: Automatically show if a specific condition is met (e.g., user is not logged in)
    // This is useful if you manually manage access on the frontend or if you bypass @login_required for partial content.
    // However, if @login_required is working, the user won't even reach this JS if they're not logged in.
    // So, this is more for "click a button to do something that requires login."

    // Method 2: Trigger when a specific action or button is clicked on the page
    // Example: If you have a "Download Report" button that requires login
    // document.getElementById('download-report-button').addEventListener('click', function(event) {
    //     event.preventDefault(); // Stop default action
    //     // Here, you'd check if the user is authenticated via AJAX or a hidden variable
    //     const isAuthenticated = /* Get this value from a Django template variable or AJAX */;
    //     if (!isAuthenticated) {
    //         showPopup();
    //     } else {
    //         // Proceed with download
    //         window.location.href = this.href;
    //     }
    // });

    // Given your current setup where @login_required redirects *the entire page*,
    // this popup is best used for actions *within* a page that requires login,
    // *not* for the initial access to the whole page.

    // If you want the popup to show up on the "model page" (instead of redirecting),
    // you would remove `@login_required` from `model_page_view` and manage authentication
    // entirely client-side using AJAX requests, or display the popup immediately if not logged in.
    // However, using @login_required for full page protection is the standard and more secure Django way.

    // If you *really* want the popup to appear *instead* of a redirect when an unauthenticated
    // user tries to access a protected page (which goes against standard Django @login_required flow),
    // you would need to modify your `auth_view` or create a new view.

    // For now, if @login_required is working, this popup would be activated by *your code*
    // for specific actions, e.g., if a user tries to click a button for a protected feature.

    // Example of triggering the popup manually for demonstration:
    // showPopup(); // Uncomment to see the popup immediately on page load for testing.
});