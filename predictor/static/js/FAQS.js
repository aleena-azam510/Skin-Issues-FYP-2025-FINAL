document.addEventListener('DOMContentLoaded', () => {
        const faqQuestions = document.querySelectorAll('.faq-question');

        faqQuestions.forEach(question => {
            question.addEventListener('click', () => {
                const faqItem = question.closest('.faq-item');
                const answer = faqItem.querySelector('.faq-answer');

                // Close any currently open answer that is not the clicked one
                document.querySelectorAll('.faq-answer.open').forEach(openAnswer => {
                    if (openAnswer !== answer) {
                        openAnswer.style.maxHeight = '0';
                        openAnswer.classList.remove('open');
                        openAnswer.closest('.faq-item').querySelector('.faq-question').classList.remove('active');
                    }
                });

                // Toggle the clicked answer
                if (answer.classList.contains('open')) {
                    answer.style.maxHeight = '0';
                    answer.classList.remove('open');
                    question.classList.remove('active');
                } else {
                    answer.style.maxHeight = answer.scrollHeight + 'px';
                    answer.classList.add('open');
                    question.classList.add('active');
                }
            });
        });
    });