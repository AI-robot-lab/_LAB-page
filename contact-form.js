/**
 * Contact Form Validation and Submission Handler
 * Laboratorium Robotów Humanoidalnych
 */

'use strict';

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('recruitmentForm');
    const studyFieldSelect = document.getElementById('studyField');
    const otherFieldGroup = document.getElementById('otherFieldGroup');
    const otherFieldInput = document.getElementById('otherField');
    
    // Show/hide "Other field" input
    if (studyFieldSelect) {
        studyFieldSelect.addEventListener('change', function() {
            if (this.value === 'inny') {
                otherFieldGroup.style.display = 'block';
                otherFieldInput.required = true;
            } else {
                otherFieldGroup.style.display = 'none';
                otherFieldInput.required = false;
                otherFieldInput.value = '';
            }
        });
    }

    // Form validation messages
    const validationMessages = {
        valueMissing: 'To pole jest wymagane',
        typeMismatch: {
            email: 'Podaj prawidłowy adres email'
        },
        patternMismatch: 'Nieprawidłowy format danych',
        rangeUnderflow: 'Wartość jest za mała',
        rangeOverflow: 'Wartość jest za duża',
        tooShort: 'Wprowadź więcej znaków',
        tooLong: 'Wprowadzono za dużo znaków'
    };

    /**
     * Validate single field
     */
    function validateField(field) {
        const errorElement = field.parentElement.querySelector('.error-message');
        
        if (!errorElement) return true;

        // Clear previous error
        errorElement.textContent = '';
        field.classList.remove('error');

        // Check validity
        if (!field.checkValidity()) {
            const validity = field.validity;
            let message = validationMessages.valueMissing;

            if (validity.typeMismatch && field.type === 'email') {
                message = validationMessages.typeMismatch.email;
            } else if (validity.patternMismatch) {
                message = validationMessages.patternMismatch;
            } else if (validity.rangeUnderflow) {
                message = `Minimalna wartość to ${field.min}`;
            } else if (validity.rangeOverflow) {
                message = `Maksymalna wartość to ${field.max}`;
            } else if (validity.tooShort) {
                message = `Minimum ${field.minLength} znaków`;
            } else if (validity.tooLong) {
                message = `Maksimum ${field.maxLength} znaków`;
            }

            errorElement.textContent = message;
            field.classList.add('error');
            return false;
        }

        return true;
    }

    /**
     * Validate radio buttons group
     */
    function validateRadioGroup(name) {
        const radios = document.querySelectorAll(`input[name="${name}"]`);
        const errorElement = document.querySelector('.team-selection + .error-message');
        
        let isChecked = false;
        radios.forEach(radio => {
            if (radio.checked) isChecked = true;
        });

        if (!isChecked && errorElement) {
            errorElement.textContent = 'Wybierz jeden z zespołów';
            return false;
        } else if (errorElement) {
            errorElement.textContent = '';
        }

        return isChecked;
    }

    /**
     * Validate checkbox
     */
    function validateCheckbox(checkbox) {
        const errorElement = checkbox.closest('.checkbox-group').querySelector('.error-message');
        
        if (!checkbox.checked) {
            errorElement.textContent = 'Musisz zaakceptować warunki';
            return false;
        } else {
            errorElement.textContent = '';
        }

        return true;
    }

    // Add real-time validation
    const inputs = form.querySelectorAll('input:not([type="radio"]):not([type="checkbox"]), select, textarea');
    inputs.forEach(input => {
        // Validate on blur
        input.addEventListener('blur', function() {
            validateField(this);
        });

        // Clear error on input
        input.addEventListener('input', function() {
            if (this.classList.contains('error')) {
                validateField(this);
            }
        });
    });

    // Validate radio buttons
    const teamRadios = document.querySelectorAll('input[name="team"]');
    teamRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            validateRadioGroup('team');
        });
    });

    // Validate GDPR checkbox
    const gdprCheckbox = document.getElementById('gdprConsent');
    if (gdprCheckbox) {
        gdprCheckbox.addEventListener('change', function() {
            validateCheckbox(this);
        });
    }

    /**
     * Form submission handler
     */
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();

            // Validate all fields
            let isValid = true;
            
            // Validate inputs
            inputs.forEach(input => {
                if (!validateField(input)) {
                    isValid = false;
                }
            });

            // Validate radio group
            if (!validateRadioGroup('team')) {
                isValid = false;
            }

            // Validate GDPR consent
            if (gdprCheckbox && !validateCheckbox(gdprCheckbox)) {
                isValid = false;
            }

            if (!isValid) {
                showMessage('Proszę poprawić błędy w formularzu', 'error');
                // Scroll to first error
                const firstError = form.querySelector('.error');
                if (firstError) {
                    firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    firstError.focus();
                }
                return;
            }

            // Submit form
            await submitForm(this);
        });
    }

    /**
     * Submit form to Formspree or send email
     */
    async function submitForm(form) {
        const submitButton = form.querySelector('button[type="submit"]');
        const formMessage = document.getElementById('formMessage');
        
        // Disable submit button
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Wysyłanie...';

        try {
            const formData = new FormData(form);
            
            // Check if Formspree is configured
            const formAction = form.getAttribute('action');
            
            if (formAction && formAction.includes('formspree.io')) {
                // Submit to Formspree
                const response = await fetch(formAction, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'Accept': 'application/json'
                    }
                });

                if (response.ok) {
                    showMessage('Dziękujemy! Twoja aplikacja została wysłana. Skontaktujemy się wkrótce.', 'success');
                    form.reset();
                    // Scroll to success message
                    formMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
                } else {
                    throw new Error('Błąd serwera');
                }
            } else {
                // Fallback: Generate mailto link
                const data = Object.fromEntries(formData);
                const emailBody = generateEmailBody(data);
                const mailtoLink = `mailto:robotlab@prz.edu.pl?subject=Aplikacja do Laboratorium - ${data.firstName} ${data.lastName}&body=${encodeURIComponent(emailBody)}`;
                
                window.location.href = mailtoLink;
                
                showMessage(
                    'Otwieramy Twojego klienta email. Jeśli nie otworzył się automatycznie, skopiuj dane i wyślij na: robotlab@prz.edu.pl', 
                    'info'
                );
            }

        } catch (error) {
            console.error('Submission error:', error);
            showMessage('Wystąpił błąd podczas wysyłania formularza. Spróbuj ponownie lub skontaktuj się bezpośrednio: robotlab@prz.edu.pl', 'error');
        } finally {
            // Re-enable submit button
            submitButton.disabled = false;
            submitButton.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Wyślij Aplikację';
        }
    }

    /**
     * Generate email body from form data
     */
    function generateEmailBody(data) {
        return `
APLIKACJA DO LABORATORIUM ROBOTÓW HUMANOIDALNYCH
================================================

DANE OSOBOWE:
- Płeć: ${data.gender}
- Imię i nazwisko: ${data.firstName} ${data.lastName}
- Email: ${data.email}

INFORMACJE AKADEMICKIE:
- Kierunek: ${data.studyField === 'inny' ? data.otherField : data.studyField}
- Poziom studiów: ${data.studyLevel}
- Semestr: ${data.semester}

DOSTĘPNOŚĆ:
- Godziny/tydzień: ${data.hoursPerWeek}
- Preferowany zespół: ${data.team}

UMIEJĘTNOŚCI:
${data.skills || 'Nie podano'}

MOTYWACJA:
${data.motivation || 'Nie podano'}

ZGODY:
- GDPR: ${data.gdprConsent ? 'TAK' : 'NIE'}

================================================
Data wysłania: ${new Date().toLocaleString('pl-PL')}
        `.trim();
    }

    /**
     * Show form message
     */
    function showMessage(message, type) {
        const formMessage = document.getElementById('formMessage');
        if (!formMessage) return;

        formMessage.textContent = message;
        formMessage.className = `form-message ${type}`;
        formMessage.style.display = 'block';

        // Auto-hide after 10 seconds for non-error messages
        if (type !== 'error') {
            setTimeout(() => {
                formMessage.style.display = 'none';
            }, 10000);
        }
    }

    // Handle reset button
    form.addEventListener('reset', function() {
        // Clear all error messages
        const errorMessages = form.querySelectorAll('.error-message');
        errorMessages.forEach(msg => msg.textContent = '');
        
        // Remove error classes
        const errorFields = form.querySelectorAll('.error');
        errorFields.forEach(field => field.classList.remove('error'));
        
        // Hide "other field" input
        otherFieldGroup.style.display = 'none';
        otherFieldInput.required = false;
        
        // Hide form message
        const formMessage = document.getElementById('formMessage');
        if (formMessage) {
            formMessage.style.display = 'none';
        }
    });
});

// Email validation helper
function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// Polish characters validation helper
function hasValidPolishCharacters(text) {
    const re = /^[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\s\-]+$/;
    return re.test(text);
}
