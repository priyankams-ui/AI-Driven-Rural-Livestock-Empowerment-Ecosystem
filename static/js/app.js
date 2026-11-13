// =========================================================
// LIVESTOCK EMPOWERMENT ECOSYSTEM
// Clean, Human-Developed Client-Side Application JavaScript
// =========================================================

// --- Application State ---
let currentUser = null; // null on page load; user must register/login
let karnatakaDistricts = [];
let allBuyerListings = [];
let allAdminLivestock = [];
let buyerCurrentPage = 1;
const BUYER_PAGE_SIZE = 6;
let currentFilteredBuyerListings = [];
let farmerCurrentPage = 1;
const FARMER_PAGE_SIZE = 5;
let currentFarmerListings = [];
let farmerSelectedImageFile = null;
let farmerDetectedDetails = null;
let farmerPredictedPriceVal = 0;
let currentBuyItem = null;
let currentAdminRemoveListing = null;

const COW_BREEDS = [
    "Ayrshire cattle", 
    "Brown Swiss cattle", 
    "Holstein Friesian cattle", 
    "Jersey cattle", 
    "Red Dane cattle"
];

const BUFFALO_BREEDS = [
    "banni", 
    "bargur", 
    "bhadwari", 
    "Chhattisgarhi", 
    "chilika", 
    "gojri", 
    "Jaffarabadi", 
    "kalahandi", 
    "luit", 
    "marathwada", 
    "mehsana", 
    "murrah", 
    "nagpuri", 
    "nili-ravi", 
    "pandharpuri", 
    "surti", 
    "toda"
];

// =========================================================
// INITIALIZATION ON PAGE LOAD
// =========================================================
document.addEventListener('DOMContentLoaded', async () => {
    // Check if user session is persisted
    const savedUser = sessionStorage.getItem('livestock_user');
    if (savedUser) {
        try {
            currentUser = JSON.parse(savedUser);
        } catch (e) {
            currentUser = null;
        }
    }

    // Fetch Karnataka administrative districts
    await fetchKarnatakaDistricts();

    // Setup initial UI & Header
    updateNavbarUserBar();

    // Check URL hash / default route (Default: HOME FIRST)
    handleInitialRoute();

    // Populate breed filters
    populateBuyerBreedDropdown();
});

// =========================================================
// 1. ROUTING & VIEW NAVIGATION
// =========================================================
function navigateTo(viewName) {
    // Protected view checks
    if (viewName === 'farmer-dash') {
        if (!currentUser || currentUser.role !== 'farmer') {
            showToast('Please login with your Farmer account to access the Farmer Dashboard.', 'warning');
            navigateTo('login');
            selectLoginRole('farmer');
            return;
        }
    } else if (viewName === 'buyer-dash') {
        if (!currentUser || currentUser.role !== 'buyer') {
            showToast('Please login with your Buyer account to access the Buyer Dashboard.', 'warning');
            navigateTo('login');
            selectLoginRole('buyer');
            return;
        }
    } else if (viewName === 'admin-dash') {
        if (!currentUser || currentUser.role !== 'admin') {
            showToast('Administrator access required.', 'error');
            navigateTo('login');
            selectLoginRole('admin');
            return;
        }
    }

    // Stop camera stream if navigating away from farmer dashboard
    if (viewName !== 'farmer-dash') {
        stopCameraStream();
    }

    // Hide all views
    document.querySelectorAll('.page-view').forEach(v => v.classList.remove('active'));

    // Show target view
    const targetEl = document.getElementById(`view-${viewName}`);
    if (targetEl) {
        targetEl.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Update public nav active styles
    document.querySelectorAll('#public-nav .nav-link').forEach(btn => btn.classList.remove('active'));
    const activeNavBtn = document.getElementById(`nav-btn-${viewName}`);
    if (activeNavBtn) activeNavBtn.classList.add('active');

    // Update hash
    window.location.hash = viewName;

    // View-specific loader triggers
    if (viewName === 'farmer-dash') {
        loadFarmerDashboardData();
    } else if (viewName === 'buyer-dash') {
        loadBuyerMarketplaceData();
    } else if (viewName === 'admin-dash') {
        loadAdminDashboardData();
    } else if (viewName === 'register') {
        checkAdminStatusForRegistration();
    }
}

function handleInitialRoute() {
    const hash = window.location.hash.replace('#', '').trim();
    if (hash && document.getElementById(`view-${hash}`)) {
        navigateTo(hash);
    } else {
        // ALWAYS SHOW HOME PAGE FIRST ON START
        navigateTo('home');
    }
}

function openActiveDashboard() {
    if (!currentUser) {
        navigateTo('login');
        return;
    }
    if (currentUser.role === 'farmer') navigateTo('farmer-dash');
    else if (currentUser.role === 'buyer') navigateTo('buyer-dash');
    else if (currentUser.role === 'admin') navigateTo('admin-dash');
    else navigateTo('home');
}

function updateNavbarUserBar() {
    const publicNav = document.getElementById('public-nav');
    const authUserBar = document.getElementById('auth-user-bar');
    const navUserName = document.getElementById('nav-user-name');
    const navUserRole = document.getElementById('nav-user-role');
    const navLoginBtn = document.getElementById('nav-btn-login');
    const navRegBtn = document.getElementById('nav-btn-register');

    if (currentUser) {
        if (navLoginBtn) navLoginBtn.style.display = 'none';
        if (navRegBtn) navRegBtn.style.display = 'none';
        if (authUserBar) authUserBar.style.display = 'flex';
        if (navUserName) navUserName.textContent = currentUser.name;
        if (navUserRole) {
            navUserRole.textContent = currentUser.role.toUpperCase();
            navUserRole.className = `user-role-tag role-${currentUser.role}`;
        }
    } else {
        if (navLoginBtn) navLoginBtn.style.display = 'inline-flex';
        if (navRegBtn) navRegBtn.style.display = 'inline-flex';
        if (authUserBar) authUserBar.style.display = 'none';
    }
}

function handleLogout() {
    stopCameraStream();
    currentUser = null;
    sessionStorage.removeItem('livestock_user');
    updateNavbarUserBar();
    showToast('Logged out successfully.', 'info');
    navigateTo('home');
}

// =========================================================
// 2. LOCATION HIERARCHY (OFFICIAL KARNATAKA LGD APIS)
// =========================================================
async function fetchKarnatakaDistricts() {
    try {
        const res = await fetch('/api/locations/districts');
        karnatakaDistricts = await res.json();

        populateDistrictSelect('form-reg-district', 'Select District');
        populateDistrictSelect('finput-district', 'Select District');
        populateDistrictSelect('bfilter-district', 'All Districts');
    } catch (e) {
        console.error("Failed to load Karnataka districts:", e);
    }
}

function populateDistrictSelect(selectId, placeholder) {
    const el = document.getElementById(selectId);
    if (!el) return;
    el.innerHTML = `<option value="">${placeholder}</option>`;
    karnatakaDistricts.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.name;
        opt.dataset.id = d.id;
        opt.textContent = d.name;
        el.appendChild(opt);
    });
}

// Cascading for Registration
async function onRegDistrictChange() {
    const distSelect = document.getElementById('form-reg-district');
    const talukSelect = document.getElementById('form-reg-taluk');
    const villageSelect = document.getElementById('form-reg-village');
    
    talukSelect.innerHTML = '<option value="">Loading taluks...</option>';
    villageSelect.innerHTML = '<option value="">Select Taluk First</option>';

    const selectedOpt = distSelect.options[distSelect.selectedIndex];
    const distId = selectedOpt.dataset.id;
    if (!distId) {
        talukSelect.innerHTML = '<option value="">Select District First</option>';
        return;
    }

    try {
        const res = await fetch(`/api/locations/districts/${distId}/taluks`);
        const taluks = await res.json();
        talukSelect.innerHTML = '<option value="">Select Taluk</option>';
        taluks.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t.name;
            opt.dataset.id = t.id;
            opt.textContent = t.name;
            talukSelect.appendChild(opt);
        });
    } catch (err) {
        talukSelect.innerHTML = '<option value="">Error loading taluks</option>';
    }
}

async function onRegTalukChange() {
    const talukSelect = document.getElementById('form-reg-taluk');
    const villageSelect = document.getElementById('form-reg-village');

    villageSelect.innerHTML = '<option value="">Loading villages...</option>';
    const selectedOpt = talukSelect.options[talukSelect.selectedIndex];
    const talukId = selectedOpt.dataset.id;
    if (!talukId) {
        villageSelect.innerHTML = '<option value="">Select Taluk First</option>';
        return;
    }

    try {
        const res = await fetch(`/api/locations/taluks/${talukId}/villages?limit=250`);
        const villages = await res.json();
        villageSelect.innerHTML = '<option value="">Select Village</option>';
        villages.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v.name;
            opt.dataset.id = v.id;
            opt.textContent = v.pincode ? `${v.name} (${v.pincode})` : v.name;
            villageSelect.appendChild(opt);
        });
    } catch (err) {
        villageSelect.innerHTML = '<option value="">Error loading villages</option>';
    }
}

// =========================================================
// 3. REGISTRATION MODULE (EMAIL OTP ENFORCED)
// =========================================================
let regResendCooldownTimer = null;
let regResendSecondsLeft = 0;
let pendingRegistrationEmail = '';
let pendingRegistrationPayload = null;

function selectRegisterRole(role) {
    if (role === 'admin') role = 'farmer'; // Admin registration is prohibited
    document.querySelectorAll('#view-register .role-pill').forEach(p => p.classList.remove('active'));
    const btn = document.getElementById(`reg-role-${role}-btn`);
    if (btn) btn.classList.add('active');

    document.getElementById('form-reg-selected-role').value = role;
}

async function handleUserRegistration(e) {
    e.preventDefault();
    const role = document.getElementById('form-reg-selected-role').value;
    const name = document.getElementById('form-reg-name').value.trim();
    const phone = document.getElementById('form-reg-phone').value.trim();
    const email = document.getElementById('form-reg-email').value.trim().toLowerCase();
    const password = document.getElementById('form-reg-password').value;
    const confirmPassword = document.getElementById('form-reg-confirm-password').value;
    const district = document.getElementById('form-reg-district').value;
    const taluk = document.getElementById('form-reg-taluk').value;
    const village = document.getElementById('form-reg-village').value;
    const address = document.getElementById('form-reg-address').value.trim();
    const pincode = document.getElementById('form-reg-pincode').value.trim();

    if (password !== confirmPassword) {
        showToast('Password and Confirm Password do not match.', 'error');
        return;
    }

    const btn = document.getElementById('btn-submit-registration');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending verification OTP...';

    const payload = {
        role,
        name,
        phone,
        email,
        password,
        confirm_password: confirmPassword,
        district,
        taluk,
        village,
        address,
        pincode
    };

    try {
        const res = await fetch('/api/auth/register-request-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Send Email Verification OTP';

        if (!res.ok) {
            showToast(data.error || 'Registration failed.', 'error');
            return;
        }

        pendingRegistrationEmail = email;
        pendingRegistrationPayload = payload;
        document.getElementById('reg-otp-email-display').textContent = email;
        document.getElementById('form-user-register').style.display = 'none';
        document.getElementById('reg-otp-step-box').style.display = 'block';
        document.getElementById('reg-otp-input').value = '';
        document.getElementById('reg-otp-input').focus();

        startRegResendCooldown(60);
        showToast(data.message || `Verification code sent to ${email}. Please check your inbox.`, 'success');

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Send Email Verification OTP';
        showToast('Server connection failed. Please try again.', 'error');
    }
}

async function handleVerifyRegOtp(e) {
    e.preventDefault();
    const otp = document.getElementById('reg-otp-input').value.trim();
    if (!otp || otp.length !== 6) {
        showToast('Please enter the 6-digit OTP code.', 'warning');
        return;
    }

    const btn = document.getElementById('btn-verify-reg-otp');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Verifying...';

    try {
        const verifyPayload = {
            ...(pendingRegistrationPayload || {}),
            email: pendingRegistrationEmail,
            otp: otp
        };

        const res = await fetch('/api/auth/register-verify-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(verifyPayload)
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-user-check"></i> Verify & Activate Account';

        if (!res.ok) {
            showToast(data.error || 'Invalid or expired OTP code.', 'error');
            return;
        }

        showToast('Account activated successfully! Please login with your credentials.', 'success');
        
        // Reset registration form
        document.getElementById('form-user-register').reset();
        backToRegForm();

        // Redirect to Login
        const role = data.user ? data.user.role : 'farmer';
        navigateTo('login');
        selectLoginRole(role);
        document.getElementById('form-login-identifier').value = pendingRegistrationEmail;

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-user-check"></i> Verify & Activate Account';
        showToast('Server error during verification.', 'error');
    }
}

async function handleResendRegOtp() {
    if (!pendingRegistrationEmail) return;
    const btn = document.getElementById('btn-resend-reg-otp');
    btn.disabled = true;

    try {
        const res = await fetch('/api/auth/resend-registration-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: pendingRegistrationEmail })
        });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || 'Could not resend OTP.', 'error');
            btn.disabled = false;
            return;
        }

        document.getElementById('reg-otp-input').value = '';
        showToast(data.message || `Verification code sent to ${pendingRegistrationEmail}. Please check your inbox.`, 'success');
        startRegResendCooldown(60);

    } catch (err) {
        btn.disabled = false;
        showToast('Failed to resend code.', 'error');
    }
}

function startRegResendCooldown(seconds) {
    regResendSecondsLeft = seconds;
    const btn = document.getElementById('btn-resend-reg-otp');
    const timerSpan = document.getElementById('reg-resend-timer');
    if (btn) btn.style.display = 'none';
    if (timerSpan) {
        timerSpan.style.display = 'inline';
        timerSpan.textContent = `Resend available in ${regResendSecondsLeft}s`;
    }

    if (regResendCooldownTimer) clearInterval(regResendCooldownTimer);
    regResendCooldownTimer = setInterval(() => {
        regResendSecondsLeft--;
        if (regResendSecondsLeft <= 0) {
            clearInterval(regResendCooldownTimer);
            if (btn) {
                btn.disabled = false;
                btn.style.display = 'inline-block';
            }
            if (timerSpan) timerSpan.style.display = 'none';
        } else {
            if (timerSpan) timerSpan.textContent = `Resend available in ${regResendSecondsLeft}s`;
        }
    }, 1000);
}

function backToRegForm() {
    document.getElementById('reg-otp-input').value = '';
    document.getElementById('reg-otp-step-box').style.display = 'none';
    document.getElementById('form-user-register').style.display = 'block';
}

function checkAdminStatusForRegistration() {
    // Admin registration is strictly prohibited in PashuDhan AI
}

// =========================================================
// 4. LOGIN MODULE (FARMER, BUYER & 2FA ADMIN)
// =========================================================
let adminResendCooldownTimer = null;
let adminResendSecondsLeft = 0;
let pendingAdminEmail = '';

function selectLoginRole(role) {
    // Hide forgot password box if open
    const forgotBox = document.getElementById('box-forgot-password');
    if (forgotBox) forgotBox.style.display = 'none';
    const rolePills = document.querySelector('#view-login .role-selector-pills');
    if (rolePills) rolePills.style.display = 'flex';
    const authHeader = document.querySelector('#view-login .auth-header');
    if (authHeader) authHeader.style.display = 'block';

    document.querySelectorAll('#view-login .role-pill').forEach(p => p.classList.remove('active'));
    const btn = document.getElementById(`login-role-${role}-btn`);
    if (btn) btn.classList.add('active');

    const userForm = document.getElementById('form-user-login');
    const adminStep1 = document.getElementById('form-admin-login-step1');
    const adminStep2 = document.getElementById('form-admin-login-step2');

    if (role === 'admin') {
        userForm.style.display = 'none';
        adminStep1.style.display = 'block';
        adminStep2.style.display = 'none';
        document.getElementById('admin-login-email').value = 'rakshitham556@gmail.com';
    } else {
        userForm.style.display = 'block';
        adminStep1.style.display = 'none';
        adminStep2.style.display = 'none';
        document.getElementById('form-login-selected-role').value = role;
        const roleLabels = {
            'farmer': 'FARMER / SELLER',
            'buyer': 'BUYER'
        };
        document.getElementById('login-role-text').textContent = roleLabels[role] || role.toUpperCase();
    }
}

function openLoginWithRole(role) {
    navigateTo('login');
    selectLoginRole(role);
}

// --- Forgot Password Flow (Admin, Farmer, Buyer) ---
let forgotResendCooldownTimer = null;
let pendingForgotEmail = '';
let pendingForgotResetToken = '';

function openForgotPasswordFlow() {
    // Hide standard forms & role selector
    document.getElementById('form-user-login').style.display = 'none';
    document.getElementById('form-admin-login-step1').style.display = 'none';
    document.getElementById('form-admin-login-step2').style.display = 'none';
    const rolePills = document.querySelector('#view-login .role-selector-pills');
    if (rolePills) rolePills.style.display = 'none';
    const authHeader = document.querySelector('#view-login .auth-header');
    if (authHeader) authHeader.style.display = 'none';

    // Show forgot password container
    const forgotBox = document.getElementById('box-forgot-password');
    forgotBox.style.display = 'block';
    document.getElementById('form-forgot-step1').style.display = 'block';
    document.getElementById('form-forgot-step2').style.display = 'none';
    document.getElementById('form-forgot-step3').style.display = 'none';
    document.getElementById('box-forgot-success').style.display = 'none';

    // Pre-fill email if user already typed one
    const userEmailInput = document.getElementById('form-login-identifier');
    const adminEmailInput = document.getElementById('admin-login-email');
    const currentTyped = (userEmailInput && userEmailInput.value) || (adminEmailInput && adminEmailInput.value) || '';
    if (currentTyped && currentTyped.includes('@')) {
        document.getElementById('forgot-email-input').value = currentTyped.trim();
    } else {
        document.getElementById('forgot-email-input').value = '';
    }
    document.getElementById('forgot-email-input').focus();
}

function cancelForgotPassword() {
    const forgotBox = document.getElementById('box-forgot-password');
    if (forgotBox) forgotBox.style.display = 'none';
    const rolePills = document.querySelector('#view-login .role-selector-pills');
    if (rolePills) rolePills.style.display = 'flex';
    const authHeader = document.querySelector('#view-login .auth-header');
    if (authHeader) authHeader.style.display = 'block';

    const currentRole = document.getElementById('form-login-selected-role') ? document.getElementById('form-login-selected-role').value : 'farmer';
    selectLoginRole(currentRole || 'farmer');
}

function backToForgotStep1() {
    document.getElementById('form-forgot-step2').style.display = 'none';
    document.getElementById('form-forgot-step1').style.display = 'block';
}

async function handleForgotRequestOtp(e) {
    e.preventDefault();
    const email = document.getElementById('forgot-email-input').value.trim().toLowerCase();
    if (!email) {
        showToast('Please enter your registered email address.', 'warning');
        return;
    }

    const btn = document.getElementById('btn-forgot-send-otp');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending OTP...';

    try {
        const res = await fetch('/api/auth/forgot-password/request-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Send Password Reset OTP';

        if (!res.ok) {
            showToast(data.error || 'Failed to dispatch reset OTP.', 'error');
            return;
        }

        pendingForgotEmail = email;
        document.getElementById('forgot-otp-email-display').textContent = email;
        document.getElementById('form-forgot-step1').style.display = 'none';
        document.getElementById('form-forgot-step2').style.display = 'block';
        document.getElementById('forgot-otp-input').value = '';
        document.getElementById('forgot-otp-input').focus();

        startForgotResendCooldown(60);
        showToast(data.message || 'Verification code sent to your email. Please check your inbox.', 'success');

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Send Password Reset OTP';
        showToast('Server connection failed.', 'error');
    }
}

async function handleForgotResendOtp() {
    if (!pendingForgotEmail) return;
    const btn = document.getElementById('btn-forgot-resend-otp');
    btn.disabled = true;

    try {
        const res = await fetch('/api/auth/forgot-password/resend-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: pendingForgotEmail })
        });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || 'Failed to resend OTP.', 'error');
            btn.disabled = false;
            return;
        }

        showToast('New password reset OTP sent to your registered email.', 'success');
        startForgotResendCooldown(60);

    } catch (err) {
        btn.disabled = false;
        showToast('Network error while resending OTP.', 'error');
    }
}

function startForgotResendCooldown(seconds) {
    let left = seconds;
    const btn = document.getElementById('btn-forgot-resend-otp');
    const timerSpan = document.getElementById('forgot-resend-timer');
    if (btn) btn.style.display = 'none';
    if (timerSpan) {
        timerSpan.style.display = 'inline';
        timerSpan.textContent = `Resend available in ${left}s`;
    }

    if (forgotResendCooldownTimer) clearInterval(forgotResendCooldownTimer);
    forgotResendCooldownTimer = setInterval(() => {
        left--;
        if (left <= 0) {
            clearInterval(forgotResendCooldownTimer);
            if (btn) {
                btn.disabled = false;
                btn.style.display = 'inline-block';
            }
            if (timerSpan) timerSpan.style.display = 'none';
        } else {
            if (timerSpan) timerSpan.textContent = `Resend available in ${left}s`;
        }
    }, 1000);
}

async function handleForgotVerifyOtp(e) {
    e.preventDefault();
    const otp = document.getElementById('forgot-otp-input').value.trim();
    if (!otp || otp.length !== 6) {
        showToast('Please enter the 6-digit OTP received on your email.', 'warning');
        return;
    }

    const btn = document.getElementById('btn-forgot-verify-otp');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Verifying...';

    try {
        const res = await fetch('/api/auth/forgot-password/verify-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: pendingForgotEmail, otp })
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Verify OTP';

        if (!res.ok) {
            showToast(data.error || 'Invalid OTP.', 'error');
            return;
        }

        pendingForgotResetToken = data.reset_token || '';
        document.getElementById('form-forgot-step2').style.display = 'none';
        document.getElementById('form-forgot-step3').style.display = 'block';
        document.getElementById('forgot-new-password').value = '';
        document.getElementById('forgot-confirm-password').value = '';
        document.getElementById('forgot-new-password').focus();
        showToast('OTP verified successfully! Please enter your new password.', 'success');

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Verify OTP';
        showToast('Server connection failed.', 'error');
    }
}

async function handleForgotResetPassword(e) {
    e.preventDefault();
    const newPass = document.getElementById('forgot-new-password').value;
    const confirmPass = document.getElementById('forgot-confirm-password').value;

    if (!newPass || !confirmPass) {
        showToast('Please enter and confirm your new password.', 'warning');
        return;
    }

    if (newPass !== confirmPass) {
        showToast('New password and confirm password do not match.', 'error');
        return;
    }

    if (newPass.length < 6) {
        showToast('Password must be at least 6 characters long.', 'error');
        return;
    }

    const btn = document.getElementById('btn-forgot-submit-reset');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Updating Password...';

    try {
        const res = await fetch('/api/auth/forgot-password/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email: pendingForgotEmail,
                reset_token: pendingForgotResetToken,
                new_password: newPass,
                confirm_password: confirmPass
            })
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Reset Password';

        if (!res.ok) {
            showToast(data.error || 'Failed to reset password.', 'error');
            return;
        }

        // Step 7: Show success message
        document.getElementById('form-forgot-step3').style.display = 'none';
        document.getElementById('box-forgot-success').style.display = 'block';
        showToast(data.message || 'Password reset successfully.', 'success');

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Reset Password';
        showToast('Server error while resetting password.', 'error');
    }
}

async function handleUserLogin(e) {
    e.preventDefault();
    const identifier = document.getElementById('form-login-identifier').value.trim();
    const password = document.getElementById('form-login-password').value;
    const role = document.getElementById('form-login-selected-role').value;

    const btn = document.getElementById('btn-submit-login');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Authenticating...';

    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ identifier, password, role })
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-right-to-bracket"></i> Login';

        if (!res.ok) {
            showToast(data.error || 'Invalid credentials.', 'error');
            return;
        }

        currentUser = data.user;
        sessionStorage.setItem('livestock_user', JSON.stringify(currentUser));
        updateNavbarUserBar();
        showToast(`Welcome back, ${currentUser.name}!`, 'success');

        // Clean form
        document.getElementById('form-login-password').value = '';

        // Role-based redirection to correct dashboard
        if (currentUser.role === 'farmer') {
            navigateTo('farmer-dash');
        } else if (currentUser.role === 'buyer') {
            navigateTo('buyer-dash');
        } else {
            navigateTo('home');
        }

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-right-to-bracket"></i> Login';
        showToast('Network error during login.', 'error');
    }
}

// Admin Step 1: Submit email & password -> Generate OTP
async function handleAdminLoginStep1(e) {
    e.preventDefault();
    const email = document.getElementById('admin-login-email').value.trim().toLowerCase();
    const password = document.getElementById('admin-login-password').value;

    const btn = document.getElementById('btn-admin-submit-step1');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Verifying credentials...';

    try {
        const res = await fetch('/api/admin/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Verify Password & Send Admin OTP';

        if (!res.ok) {
            showToast(data.error || 'Administrator verification failed.', 'error');
            return;
        }

        pendingAdminEmail = email;
        document.getElementById('admin-otp-target-email').textContent = email;
        document.getElementById('form-admin-login-step1').style.display = 'none';
        document.getElementById('form-admin-login-step2').style.display = 'block';
        document.getElementById('admin-otp-input').value = '';
        document.getElementById('admin-otp-input').focus();

        startAdminResendCooldown(60);
        showToast(data.message || 'Two-factor OTP dispatched to admin email. Please check your inbox.', 'success');

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Verify Password & Send Admin OTP';
        showToast('Server error during admin authentication.', 'error');
    }
}

// Admin Step 2: Verify 6-digit OTP -> Create admin session
async function handleAdminVerifyOtp(e) {
    e.preventDefault();
    const otp = document.getElementById('admin-otp-input').value.trim();
    if (!otp || otp.length !== 6) {
        showToast('Please enter the 6-digit admin OTP.', 'warning');
        return;
    }

    const btn = document.getElementById('btn-admin-submit-step2');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Authenticating...';

    try {
        const res = await fetch('/api/admin/verify-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: pendingAdminEmail, otp })
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-unlock-keyhole"></i> Authenticate & Open Console';

        if (!res.ok) {
            showToast(data.error || 'Admin verification failed.', 'error');
            return;
        }

        currentUser = data.user;
        sessionStorage.setItem('livestock_user', JSON.stringify(currentUser));
        updateNavbarUserBar();
        showToast(`Administrator authenticated! Welcome Rakshitha M.`, 'success');

        // Reset forms
        document.getElementById('admin-login-password').value = '';
        document.getElementById('admin-otp-input').value = '';

        navigateTo('admin-dash');

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-unlock-keyhole"></i> Authenticate & Open Console';
        showToast('Server error during OTP verification.', 'error');
    }
}

async function handleResendAdminOtp() {
    if (!pendingAdminEmail) return;
    const btn = document.getElementById('btn-resend-admin-otp');
    btn.disabled = true;

    try {
        const res = await fetch('/api/admin/resend-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: pendingAdminEmail })
        });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || 'Failed to resend admin OTP.', 'error');
            btn.disabled = false;
            return;
        }

        showToast('A new OTP has been dispatched to rakshitham556@gmail.com.', 'success');
        startAdminResendCooldown(60);

    } catch (err) {
        btn.disabled = false;
        showToast('Network error while resending admin OTP.', 'error');
    }
}

function startAdminResendCooldown(seconds) {
    adminResendSecondsLeft = seconds;
    const btn = document.getElementById('btn-resend-admin-otp');
    const timerSpan = document.getElementById('admin-resend-timer');
    if (btn) btn.style.display = 'none';
    if (timerSpan) {
        timerSpan.style.display = 'inline';
        timerSpan.textContent = `Resend available in ${adminResendSecondsLeft}s`;
    }

    if (adminResendCooldownTimer) clearInterval(adminResendCooldownTimer);
    adminResendCooldownTimer = setInterval(() => {
        adminResendSecondsLeft--;
        if (adminResendSecondsLeft <= 0) {
            clearInterval(adminResendCooldownTimer);
            if (btn) {
                btn.disabled = false;
                btn.style.display = 'inline-block';
            }
            if (timerSpan) timerSpan.style.display = 'none';
        } else {
            if (timerSpan) timerSpan.textContent = `Resend available in ${adminResendSecondsLeft}s`;
        }
    }, 1000);
}

function backToAdminStep1() {
    document.getElementById('form-admin-login-step2').style.display = 'none';
    document.getElementById('form-admin-login-step1').style.display = 'block';
}

// =========================================================
// 5. FARMER DASHBOARD & WORKFLOW
// =========================================================
function switchFarmerTab(tabName) {
    document.querySelectorAll('.farmer-subtab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.dash-sidebar .side-nav-btn').forEach(b => b.classList.remove('active'));

    const tabEl = document.getElementById(`ftab-${tabName}`);
    if (tabEl) tabEl.classList.add('active');

    const navBtn = document.getElementById(`fnav-${tabName}`);
    if (navBtn) navBtn.classList.add('active');

    if (tabName === 'add-livestock') {
        if (activePhotoSourceMode === 'camera') {
            const reviewBox = document.getElementById('farmer-camera-review');
            if (!reviewBox || reviewBox.style.display === 'none') {
                startCameraStream();
            }
        }
    } else {
        stopCameraStream();
    }

    if (tabName === 'overview' || tabName === 'listings') {
        loadFarmerListings();
    } else if (tabName === 'history') {
        loadFarmerHistory();
    } else if (tabName === 'sold') {
        loadFarmerSold();
    } else if (tabName === 'inquiries') {
        loadFarmerInquiries();
    } else if (tabName === 'profile') {
        renderFarmerProfile();
    }
}

function loadFarmerDashboardData() {
    if (!currentUser) return;
    document.getElementById('farmer-dash-username').textContent = currentUser.name;
    document.getElementById('farmer-welcome-name').textContent = currentUser.name;
    document.getElementById('farmer-dash-userloc').textContent = `${currentUser.district || 'Karnataka'}, India`;
    switchFarmerTab('overview');
}

function renderFarmerProfile() {
    if (!currentUser) return;
    document.getElementById('fprof-name').textContent = currentUser.name;
    document.getElementById('fprof-email').textContent = currentUser.email;
    document.getElementById('fprof-phone').textContent = currentUser.phone;
    document.getElementById('fprof-district').textContent = currentUser.district || 'Mandya';
    document.getElementById('fprof-taluk').textContent = currentUser.taluk || 'Maddur';
    document.getElementById('fprof-village').textContent = currentUser.village || 'KM Doddi';
    document.getElementById('fprof-address').textContent = currentUser.address || '--';
    document.getElementById('fprof-pincode').textContent = currentUser.pincode || '--';
}

// --- Live Camera & Photo Source Controllers ---
let activeCameraStream = null;
let currentFacingMode = 'environment'; // 'environment' (rear) or 'user' (front)
let activePhotoSourceMode = 'camera'; // 'camera' or 'gallery'
let capturedLivePhotoBlob = null;

function switchPhotoSource(mode) {
    activePhotoSourceMode = mode;
    const tabCam = document.getElementById('tab-source-camera');
    const tabGal = document.getElementById('tab-source-gallery');
    const camWrap = document.getElementById('farmer-camera-wrapper');
    const galWrap = document.getElementById('farmer-gallery-wrapper');

    if (mode === 'camera') {
        if (tabCam) tabCam.classList.add('active');
        if (tabGal) tabGal.classList.remove('active');
        if (camWrap) camWrap.style.display = 'block';
        if (galWrap) galWrap.style.display = 'none';

        const reviewBox = document.getElementById('farmer-camera-review');
        if (!reviewBox || reviewBox.style.display === 'none') {
            startCameraStream();
        }
    } else {
        if (tabGal) tabGal.classList.add('active');
        if (tabCam) tabCam.classList.remove('active');
        if (camWrap) camWrap.style.display = 'none';
        if (galWrap) galWrap.style.display = 'block';
        stopCameraStream();
    }
}

async function startCameraStream() {
    const video = document.getElementById('farmer-camera-video');
    const statusLabel = document.getElementById('camera-status-label');
    const alertBadge = document.getElementById('camera-live-indicator-badge');
    const alertBox = document.getElementById('camera-permission-alert');

    if (!video) return;
    stopCameraStream();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        if (alertBox) {
            alertBox.style.display = 'block';
            alertBox.innerHTML = `
                <div class="alert-box alert-warning">
                    <i class="fa-solid fa-camera"></i>
                    <div>
                        <strong>To use the real live camera:</strong>
                        <p class="mb-1">Browsers require accessing via Localhost to enable live camera streaming.</p>
                        <div class="mt-2">
                            <a href="http://localhost:5000/#farmer-dash" class="btn btn-sm btn-primary" style="text-decoration:none;">
                                <i class="fa-solid fa-camera"></i> Open on http://localhost:5000
                            </a>
                        </div>
                    </div>
                </div>
            `;
        }
        return;
    }

    try {
        if (alertBox) alertBox.style.display = 'none';

        const constraints = {
            video: {
                facingMode: { ideal: currentFacingMode },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (strictErr) {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }

        activeCameraStream = stream;
        video.srcObject = stream;
        await video.play();

        if (statusLabel) {
            statusLabel.textContent = (currentFacingMode === 'environment') ? 'Rear Camera Active' : 'Camera Active';
        }
        if (alertBadge) {
            alertBadge.style.display = 'inline-flex';
        }
    } catch (err) {
        console.warn('Camera stream error:', err);
        stopCameraStream();
        if (alertBox) {
            alertBox.style.display = 'block';
            alertBox.innerHTML = `
                <div class="alert-box alert-warning">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <div>
                        <strong>Camera Permission Required:</strong>
                        <p class="mb-1">Please allow camera access in your browser site permissions to enable the live camera viewfinder.</p>
                        <button type="button" class="btn btn-sm btn-primary mt-1" onclick="startCameraStream()">
                            <i class="fa-solid fa-rotate-right"></i> Allow & Retry Camera
                        </button>
                    </div>
                </div>
            `;
        }
    }
}

function stopCameraStream() {
    if (activeCameraStream) {
        activeCameraStream.getTracks().forEach(track => {
            try { track.stop(); } catch (e) {}
        });
        activeCameraStream = null;
    }
    const video = document.getElementById('farmer-camera-video');
    if (video) {
        video.srcObject = null;
    }
    const alertBadge = document.getElementById('camera-live-indicator-badge');
    if (alertBadge) alertBadge.style.display = 'none';
}

function flipCameraFacing() {
    currentFacingMode = (currentFacingMode === 'environment') ? 'user' : 'environment';
    startCameraStream();
}

function captureLivePhoto() {
    const video = document.getElementById('farmer-camera-video');
    const canvas = document.getElementById('farmer-camera-canvas');

    if (activeCameraStream && video && video.videoWidth > 0 && video.videoHeight > 0) {
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        canvas.width = vw;
        canvas.height = vh;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, vw, vh);

        canvas.toBlob(blob => {
            if (!blob) {
                showToast('Failed to snapshot camera photo.', 'error');
                return;
            }
            capturedLivePhotoBlob = blob;
            const filename = `camera_capture_${Date.now()}.jpg`;
            farmerSelectedImageFile = new File([blob], filename, { type: 'image/jpeg' });

            const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
            const previewImg = document.getElementById('farmer-captured-preview-img');
            if (previewImg) previewImg.src = dataUrl;

            const vf = document.getElementById('farmer-camera-viewfinder');
            const rev = document.getElementById('farmer-camera-review');
            if (vf) vf.style.display = 'none';
            if (rev) rev.style.display = 'block';

            stopCameraStream();
        }, 'image/jpeg', 0.92);
    } else {
        showToast('Real camera is not currently streaming. Please click Retry Camera or open on Localhost.', 'warning');
        startCameraStream();
    }
}

function retakeLivePhoto() {
    capturedLivePhotoBlob = null;
    farmerSelectedImageFile = null;
    farmerDetectedDetails = null;
    farmerPredictedPriceVal = 0;

    const vf = document.getElementById('farmer-camera-viewfinder');
    const rev = document.getElementById('farmer-camera-review');
    const statusBox = document.getElementById('detection-status-box');
    const rejBanner = document.getElementById('detection-rejection-banner');
    const successCard = document.getElementById('detection-success-card');
    const priceBox = document.getElementById('predicted-price-result-box');

    if (vf) vf.style.display = 'block';
    if (rev) rev.style.display = 'none';
    if (statusBox) statusBox.style.display = 'none';
    if (rejBanner) rejBanner.style.display = 'none';
    if (successCard) successCard.style.display = 'none';
    if (priceBox) priceBox.style.display = 'none';

    // Reset detected fields
    const gInput = document.getElementById('finput-gender');
    if (gInput) gInput.value = '';
    const gText = document.getElementById('fdisplay-gender-text');
    if (gText) gText.textContent = '--';
    const detG = document.getElementById('det-gender');
    if (detG) detG.textContent = '--';
    const detGC = document.getElementById('det-gender-conf');
    if (detGC) detGC.textContent = '--';

    if (activePhotoSourceMode === 'camera') {
        startCameraStream();
    }
}

async function confirmAndDetectLivePhoto() {
    if (!farmerSelectedImageFile) {
        showToast('Please capture or choose a photo first.', 'warning');
        return;
    }
    const btn = document.getElementById('btn-confirm-live-detect');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Detecting livestock details...';
    }
    try {
        await runImageDetection();
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Confirm & Detect Details';
        }
    }
}

// --- Farmer Studio Workflow ---
function triggerFarmerFileInput() {
    document.getElementById('farmer-image-file').click();
}

function handleFarmerImageChosen(e) {
    const file = e.target.files[0];
    if (!file) return;

    farmerSelectedImageFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
        document.getElementById('farmer-img-preview').src = event.target.result;
        document.getElementById('farmer-img-preview-box').style.display = 'block';
        document.getElementById('dropzone-placeholder').style.display = 'none';
        document.getElementById('detect-action-row').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearFarmerImage(e) {
    if (e) e.stopPropagation();
    farmerSelectedImageFile = null;
    capturedLivePhotoBlob = null;
    farmerDetectedDetails = null;
    farmerPredictedPriceVal = 0;
    document.getElementById('farmer-image-file').value = '';
    document.getElementById('farmer-img-preview').src = '';
    document.getElementById('farmer-img-preview-box').style.display = 'none';
    document.getElementById('dropzone-placeholder').style.display = 'block';
    document.getElementById('detect-action-row').style.display = 'none';
    document.getElementById('detection-status-box').style.display = 'none';
    document.getElementById('predicted-price-result-box').style.display = 'none';

    // Reset detected gender fields (do not default to Female)
    const gInput = document.getElementById('finput-gender');
    if (gInput) gInput.value = '';
    const gText = document.getElementById('fdisplay-gender-text');
    if (gText) gText.textContent = '--';
    const detG = document.getElementById('det-gender');
    if (detG) detG.textContent = '--';
    const detGC = document.getElementById('det-gender-conf');
    if (detGC) detGC.textContent = '--';
    const lactSelect = document.getElementById('finput-lactation');
    if (lactSelect) lactSelect.value = '1st';
    const calfSelect = document.getElementById('finput-calf');
    if (calfSelect) calfSelect.value = '1st';
}

async function runImageDetection() {
    if (!farmerSelectedImageFile) {
        showToast('Please capture or upload a photo first.', 'warning');
        return;
    }

    const btn = document.getElementById('btn-detect-image');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Detecting livestock details...';
    }

    const formData = new FormData();
    formData.append('file', farmerSelectedImageFile);

    try {
        const res = await fetch('/api/predict/image-analysis', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> DETECT DETAILS';
        }
        document.getElementById('detection-status-box').style.display = 'block';

        if (!data.is_valid || data.status === 'rejected' || data.species === 'Unknown' || !['Cow', 'Buffalo'].includes(data.species)) {
            document.getElementById('detection-rejection-banner').style.display = 'flex';
            document.getElementById('detection-success-card').style.display = 'none';
            document.getElementById('predicted-price-result-box').style.display = 'none';
            farmerDetectedDetails = null;
            const rejTitle = document.getElementById('detection-rejection-title');
            if (rejTitle && data.error) rejTitle.textContent = data.error;
            showToast(data.error || 'No valid cow or buffalo detected. Please capture a clear livestock image.', 'error');
            return;
        }

        // Detection Success (Cow or Buffalo only)
        document.getElementById('detection-rejection-banner').style.display = 'none';
        document.getElementById('detection-success-card').style.display = 'block';

        document.getElementById('det-species').textContent = data.species;
        const spConfElem = document.getElementById('det-species-conf');
        if (spConfElem) spConfElem.textContent = data.species_confidence ? `${(data.species_confidence * 100).toFixed(1)}%` : '--';
        document.getElementById('det-breed').textContent = data.breed;
        const brConfElem = document.getElementById('det-breed-conf');
        if (brConfElem) brConfElem.textContent = data.breed_confidence ? `${(data.breed_confidence * 100).toFixed(1)}%` : '--';

        const detectedGender = data.gender;
        const genderElem = document.getElementById('det-gender');
        if (genderElem) genderElem.textContent = detectedGender || '--';
        const genderConfElem = document.getElementById('det-gender-conf');
        if (genderConfElem) genderConfElem.textContent = data.gender_confidence ? `${(data.gender_confidence * 100).toFixed(1)}%` : '--';

        farmerDetectedDetails = data;
        
        if (detectedGender === 'Male' || detectedGender === 'Female') {
            // Auto-assign detected gender in Step 2 form (no manual gender selection)
            const gInput = document.getElementById('finput-gender');
            if (gInput) {
                gInput.value = detectedGender;
            }
            const gText = document.getElementById('fdisplay-gender-text');
            if (gText) {
                gText.textContent = detectedGender;
            }
            onFarmerGenderChanged();

            const pSelect = document.getElementById('finput-pregnancy');
            if (pSelect) {
                pSelect.value = 'No';
                onFarmerPregnancyChanged();
            }

            showToast(`Livestock identified: ${data.species} (${data.breed}), Detected Gender: ${detectedGender}.`, 'success');
        } else {
            // Gender is Uncertain
            showToast(data.gender_message || "Gender could not be determined reliably. Please upload a clearer full-body image showing the animal's relevant features.", 'warning');
        }

        // Pre-fill location if available
        if (currentUser && currentUser.district) {
            const dSelect = document.getElementById('finput-district');
            dSelect.value = currentUser.district;
            await onFarmerStudioDistrictChange();
            if (currentUser.taluk) {
                document.getElementById('finput-taluk').value = currentUser.taluk;
                await onFarmerStudioTalukChange();
                if (currentUser.village) {
                    document.getElementById('finput-village').value = currentUser.village;
                }
            }
        }
    } catch (err) {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> DETECT DETAILS';
        }
        showToast('Detection service failed.', 'error');
    }
}

async function onFarmerStudioDistrictChange() {
    const dSelect = document.getElementById('finput-district');
    const tSelect = document.getElementById('finput-taluk');
    const vSelect = document.getElementById('finput-village');

    tSelect.innerHTML = '<option value="">Loading taluks...</option>';
    vSelect.innerHTML = '<option value="">Select Taluk First</option>';

    const opt = dSelect.options[dSelect.selectedIndex];
    const distId = opt.dataset.id;
    if (!distId) {
        tSelect.innerHTML = '<option value="">Select District First</option>';
        return;
    }

    try {
        const res = await fetch(`/api/locations/districts/${distId}/taluks`);
        const taluks = await res.json();
        tSelect.innerHTML = '<option value="">Select Taluk</option>';
        taluks.forEach(t => {
            const o = document.createElement('option');
            o.value = t.name;
            o.dataset.id = t.id;
            o.textContent = t.name;
            tSelect.appendChild(o);
        });
    } catch (e) {
        tSelect.innerHTML = '<option value="">Error loading taluks</option>';
    }
    onFarmerInputsChanged();
}

async function onFarmerStudioTalukChange() {
    const tSelect = document.getElementById('finput-taluk');
    const vSelect = document.getElementById('finput-village');

    vSelect.innerHTML = '<option value="">Loading villages...</option>';
    const opt = tSelect.options[tSelect.selectedIndex];
    const talukId = opt.dataset.id;
    if (!talukId) {
        vSelect.innerHTML = '<option value="">Select Taluk First</option>';
        return;
    }

    try {
        const res = await fetch(`/api/locations/taluks/${talukId}/villages?limit=250`);
        const villages = await res.json();
        vSelect.innerHTML = '<option value="">Select Village</option>';
        villages.forEach(v => {
            const o = document.createElement('option');
            o.value = v.name;
            o.dataset.id = v.id;
            o.textContent = v.name;
            vSelect.appendChild(o);
        });
    } catch (e) {
        vSelect.innerHTML = '<option value="">Error loading villages</option>';
    }
    onFarmerInputsChanged();
}

function onFarmerInputsChanged() {
    // Hide price card if input changes to mandate clicking "PREDICT PRICE"
    document.getElementById('predicted-price-result-box').style.display = 'none';
    document.getElementById('predict-val-error').style.display = 'none';
}

function onFarmerLactationChanged() {
    const lactSelect = document.getElementById('finput-lactation');
    const calfSelect = document.getElementById('finput-calf');
    const milkInput = document.getElementById('finput-milk');
    const milkHelp = document.getElementById('milk-help-text');

    if (lactSelect && calfSelect) {
        calfSelect.value = lactSelect.value;
    }

    if (lactSelect && milkInput) {
        const lact = lactSelect.value;
        if (lact === '1st' || lact === '2nd' || lact === '3rd') {
            milkInput.max = '30';
            milkInput.placeholder = 'Enter litres/day (1 to 30 L/day)';
            if (milkHelp) milkHelp.textContent = 'Milk production for ' + lact + ' lactation (Max 30 L/day)';
            if (parseFloat(milkInput.value) > 30) {
                milkInput.value = '30';
            }
        } else if (lact === '4th') {
            milkInput.max = '15';
            milkInput.placeholder = 'Enter litres/day (Max 15 L/day for 4th lactation)';
            if (milkHelp) milkHelp.textContent = 'Milk production is restricted to maximum 15 L/day for 4th lactation';
            if (parseFloat(milkInput.value) > 15) {
                milkInput.value = '15';
            }
        } else if (lact === '5th') {
            milkInput.max = '10';
            milkInput.placeholder = 'Enter litres/day (Max 10 L/day for 5th lactation)';
            if (milkHelp) milkHelp.textContent = 'Milk production is restricted to maximum 10 L/day for 5th lactation';
            if (parseFloat(milkInput.value) > 10) {
                milkInput.value = '10';
            }
        }
    }
    onFarmerInputsChanged();
}
window.onFarmerLactationChanged = onFarmerLactationChanged;

function onFarmerGenderChanged() {
    const gender = document.getElementById('finput-gender').value;
    const pregGroup = document.getElementById('group-pregnancy');
    const pregMonthRow = document.getElementById('row-pregnancy-month');
    const lactationCalfRow = document.getElementById('row-lactation-calf');
    const milkInput = document.getElementById('finput-milk');
    const milkHelp = document.getElementById('milk-help-text');

    if (gender === 'Male') {
        if (pregGroup) pregGroup.style.display = 'none';
        if (pregMonthRow) pregMonthRow.style.display = 'none';
        if (lactationCalfRow) lactationCalfRow.style.display = 'none';
        const milkRow = document.getElementById('row-milk');
        if (milkRow) milkRow.style.display = 'none';
        if (milkInput) {
            milkInput.value = '0';
            milkInput.disabled = true;
        }
        if (milkHelp) milkHelp.textContent = 'Milk production is not applicable for male animals (0 L/day)';
    } else {
        if (pregGroup) pregGroup.style.display = 'block';
        if (lactationCalfRow) lactationCalfRow.style.display = 'flex';
        const milkRow = document.getElementById('row-milk');
        if (milkRow) milkRow.style.display = 'flex';
        const pregVal = document.getElementById('finput-pregnancy') ? document.getElementById('finput-pregnancy').value : 'No';
        if (pregMonthRow) pregMonthRow.style.display = (pregVal === 'Yes') ? 'flex' : 'none';
        if (milkInput) {
            milkInput.disabled = false;
            if (milkInput.value === '0') milkInput.value = '';
        }
        if (milkHelp) milkHelp.textContent = 'Enter 0 for bullocks or dry cattle';
    }
    onFarmerInputsChanged();
}

function onFarmerPregnancyChanged() {
    const pregVal = document.getElementById('finput-pregnancy').value;
    const pregMonthRow = document.getElementById('row-pregnancy-month');
    if (pregMonthRow) {
        pregMonthRow.style.display = (pregVal === 'Yes') ? 'flex' : 'none';
    }
    onFarmerInputsChanged();
}

async function triggerFarmerPricePrediction() {
    const errBox = document.getElementById('predict-val-error');
    errBox.style.display = 'none';

    if (!farmerDetectedDetails || !farmerDetectedDetails.is_valid) {
        errBox.textContent = 'Please upload a photo and detect species/breed first.';
        errBox.style.display = 'block';
        return;
    }

    const gender = document.getElementById('finput-gender').value;
    const pregnancy = (gender === 'Female') ? document.getElementById('finput-pregnancy').value : 'No';
    const pregnancyMonth = (gender === 'Female' && pregnancy === 'Yes') 
        ? (parseInt(document.getElementById('finput-pregnancy-month').value) || 1) 
        : 0;

    let milk = document.getElementById('finput-milk').value.trim();
    if (gender === 'Male') {
        milk = '0';
    }

    const age = document.getElementById('finput-age').value.trim();
    const district = document.getElementById('finput-district').value;
    const taluk = document.getElementById('finput-taluk').value;
    const village = document.getElementById('finput-village').value;

    if (gender === 'Female' && milk === '') {
        errBox.textContent = 'Please enter daily milk production.';
        errBox.style.display = 'block';
        return;
    }

    const lactation = (gender === 'Female') 
        ? (document.getElementById('finput-lactation') ? document.getElementById('finput-lactation').value : '1st')
        : 'Not Applicable';
    const calf = (gender === 'Female') 
        ? (document.getElementById('finput-calf') ? document.getElementById('finput-calf').value : '1st')
        : 'Not Applicable';

    if (gender === 'Female') {
        const milkVal = parseFloat(milk);
        if (lactation === '5th' && milkVal > 10) {
            errBox.textContent = 'For 5th lactation, milk production cannot exceed 10 L/day.';
            errBox.style.display = 'block';
            return;
        }
        if (lactation === '4th' && milkVal > 15) {
            errBox.textContent = 'For 4th lactation, milk production cannot exceed 15 L/day.';
            errBox.style.display = 'block';
            return;
        }
        if ((lactation === '1st' || lactation === '2nd' || lactation === '3rd') && milkVal > 30) {
            errBox.textContent = `For ${lactation} lactation, milk production cannot exceed 30 L/day.`;
            errBox.style.display = 'block';
            return;
        }
    }
    if (age === '') {
        errBox.textContent = 'Please enter age of the livestock.';
        errBox.style.display = 'block';
        return;
    }
    if (!district || district === '') {
        errBox.textContent = 'Please select district.';
        errBox.style.display = 'block';
        return;
    }
    if (!taluk || taluk === '') {
        errBox.textContent = 'Please select taluk.';
        errBox.style.display = 'block';
        return;
    }
    if (!village || village === '') {
        errBox.textContent = 'Please select village.';
        errBox.style.display = 'block';
        return;
    }

    const btn = document.getElementById('btn-predict-price');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Predicting price...';

    const payload = {
        species: farmerDetectedDetails.species,
        breed: farmerDetectedDetails.breed,
        gender: gender,
        pregnancy: pregnancy,
        is_pregnant: (pregnancy === 'Yes') ? 1 : 0,
        pregnancy_month: pregnancyMonth,
        lactation_number: lactation,
        calf_number: calf,
        health_score: (farmerDetectedDetails && farmerDetectedDetails.health_score) ? farmerDetectedDetails.health_score : 4,
        milk_production: parseFloat(milk) || 0,
        age_years: parseFloat(age),
        district: district,
        taluk: taluk,
        village: village
    };

    try {
        const res = await fetch('/api/predict/price', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-calculator"></i> PREDICT PRICE';

        if (!res.ok) {
            errBox.textContent = data.error || 'Price calculation failed.';
            errBox.style.display = 'block';
            return;
        }

        farmerPredictedPriceVal = data.predicted_price;
        farmerDetectedDetails.gender = gender;
        farmerDetectedDetails.pregnancy = pregnancy;
        farmerDetectedDetails.pregnancy_month = pregnancyMonth;

        document.getElementById('predicted-price-val').textContent = data.predicted_price.toLocaleString('en-IN');
        document.getElementById('predicted-price-range').textContent = `Calibrated Fair Range (85% Conf): ₹${data.price_range_min.toLocaleString('en-IN')} to ₹${data.price_range_max.toLocaleString('en-IN')}`;
        
        // Confidence badge
        const badgeEl = document.getElementById('predicted-confidence-badge');
        if (badgeEl) {
            const conf = data.confidence || 'High';
            if (conf === 'High') {
                badgeEl.style.backgroundColor = '#dcfce7';
                badgeEl.style.color = '#15803d';
                badgeEl.innerHTML = '<i class="fa-solid fa-circle-check"></i> High Evidence';
            } else if (conf === 'Moderate') {
                badgeEl.style.backgroundColor = '#fef9c3';
                badgeEl.style.color = '#854d0e';
                badgeEl.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Moderate Evidence';
            } else {
                badgeEl.style.backgroundColor = '#fee2e2';
                badgeEl.style.color = '#991b1b';
                badgeEl.innerHTML = '<i class="fa-solid fa-circle-exclamation"></i> Low Evidence (Inspect)';
            }
        }

        // Valuation basis note
        const basisEl = document.getElementById('predicted-price-basis');
        if (basisEl) {
            basisEl.innerHTML = `<i class="fa-solid fa-scale-balanced"></i> ${data.valuation_basis || 'Learned from Karnataka market transactions and NABARD SLUCC unit cost schedules.'}`;
        }
        
        // Summary text
        let pregText = '';
        let lactCalfText = '';
        if (gender === 'Female') {
            pregText = (pregnancy === 'Yes') ? ` | <strong>Pregnancy:</strong> Month ${pregnancyMonth}` : ' | <strong>Pregnancy:</strong> No';
            lactCalfText = ` | <strong>Lactation:</strong> ${lactation} | <strong>Calf:</strong> ${calf}`;
        }
        document.getElementById('price-result-summary-text').innerHTML = `
            <div><strong>Animal:</strong> ${farmerDetectedDetails.species} (${farmerDetectedDetails.breed}) | <strong>Gender:</strong> ${gender}${pregText}${lactCalfText}</div>
            <div><strong>Milk:</strong> ${milk} L/day | <strong>Age:</strong> ${age} Yrs | <strong>Location:</strong> ${village}, ${taluk}, ${district}</div>
        `;

        // Pre-fill asking price & address
        document.getElementById('finput-asking-price').value = data.predicted_price;
        if (currentUser) {
            document.getElementById('finput-address').value = currentUser.address || '';
            document.getElementById('finput-pincode').value = currentUser.pincode || '';
        }

        document.getElementById('predicted-price-result-box').style.display = 'block';
        showToast('Price predicted successfully! Enter pickup address and publish.', 'success');

    } catch (e) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-calculator"></i> PREDICT PRICE';
        errBox.textContent = 'Server error during price prediction.';
        errBox.style.display = 'block';
    }
}

async function handlePublishLivestock(e) {
    e.preventDefault();
    if (!farmerDetectedDetails || !farmerPredictedPriceVal) {
        showToast('Please complete detection and price prediction first.', 'warning');
        return;
    }

    const askingPrice = parseInt(document.getElementById('finput-asking-price').value);
    const address = document.getElementById('finput-address').value.trim();
    const pincode = document.getElementById('finput-pincode').value.trim();
    const milk = parseFloat(document.getElementById('finput-milk').value);
    const age = parseFloat(document.getElementById('finput-age').value);
    const district = document.getElementById('finput-district').value;
    const taluk = document.getElementById('finput-taluk').value;
    const village = document.getElementById('finput-village').value;

    const btn = document.getElementById('btn-publish-livestock');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Publishing listing...';

    const lactation = (farmerDetectedDetails.gender === 'Female') 
        ? (document.getElementById('finput-lactation') ? document.getElementById('finput-lactation').value : '1st')
        : 'Not Applicable';
    const calf = (farmerDetectedDetails.gender === 'Female') 
        ? (document.getElementById('finput-calf') ? document.getElementById('finput-calf').value : '1st')
        : 'Not Applicable';

    if (farmerDetectedDetails.gender === 'Female') {
        if (lactation === '5th' && milk > 10) {
            showToast('For 5th lactation, milk production cannot exceed 10 L/day.', 'error');
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> PUBLISH LIVESTOCK';
            return;
        }
        if (lactation === '4th' && milk > 15) {
            showToast('For 4th lactation, milk production cannot exceed 15 L/day.', 'error');
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> PUBLISH LIVESTOCK';
            return;
        }
    }

    const payload = {
        farmer_id: currentUser.id,
        farmer_name: currentUser.name,
        farmer_phone: currentUser.phone,
        image_url: farmerDetectedDetails.image_url,
        species: farmerDetectedDetails.species,
        breed: farmerDetectedDetails.breed,
        gender: farmerDetectedDetails.gender,
        health_score: farmerDetectedDetails.health_score || 4,
        health_text: farmerDetectedDetails.health_condition_text || 'Good',
        age_years: age,
        lactation_number: lactation,
        calf_number: calf,
        milk_production: milk,
        district: district,
        taluk: taluk,
        village: village,
        address: address,
        pincode: pincode,
        predicted_price: farmerPredictedPriceVal,
        asking_price: askingPrice
    };

    try {
        const res = await fetch('/api/listings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> PUBLISH LIVESTOCK';

        if (!res.ok) {
            showToast(data.error || 'Failed to publish listing.', 'error');
            return;
        }

        showToast('Livestock published successfully to marketplace!', 'success');
        clearFarmerImage();
        document.getElementById('form-publish-livestock').reset();
        switchFarmerTab('listings');

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> PUBLISH LIVESTOCK';
        showToast('Server error during publication.', 'error');
    }
}

// --- Farmer Listings, History & Sold Loaders ---
async function loadFarmerListings() {
    if (!currentUser) return;
    try {
        const res = await fetch(`/api/farmer/${currentUser.id}/listings`);
        const listings = await res.json();

        // Update metric counts
        const available = listings.filter(l => l.status === 'Available').length;
        const sold = listings.filter(l => l.status === 'SOLD_OUT').length;
        document.getElementById('fmetric-total').textContent = listings.length;
        document.getElementById('fmetric-available').textContent = available;
        document.getElementById('fmetric-sold').textContent = sold;

        // Render Recent Table (Overview)
        const recentTbody = document.getElementById('farmer-recent-tbody');
        if (recentTbody) {
            if (listings.length === 0) {
                recentTbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No livestock listed yet. Click "Add New Livestock" above.</td></tr>';
            } else {
                recentTbody.innerHTML = listings.slice(0, 5).map(l => `
                    <tr>
                        <td><img src="${l.image_url}" class="table-thumb" alt="Thumb"></td>
                        <td><strong>${l.species}</strong></td>
                        <td>${l.breed}</td>
                        <td>${l.gender}</td>
                        <td>${l.milk_production} L/day</td>
                        <td><strong>₹${l.asking_price.toLocaleString('en-IN')}</strong></td>
                        <td><span class="badge-status ${l.status === 'Available' ? 'status-available' : 'status-sold'}">${formatDisplayStatus(l.status, 'farmer')}</span></td>
                    </tr>
                `).join('');
            }
        }

        // Render Full Listings Table with Pagination
        currentFarmerListings = listings || [];
        farmerCurrentPage = 1;
        renderFarmerListingsTable();
    } catch (e) {
        console.error(e);
    }
}

function renderFarmerListingsTable() {
    const fullTbody = document.getElementById('farmer-listings-tbody');
    const paginationEl = document.getElementById('farmer-listings-pagination');
    if (!fullTbody) return;

    if (!currentFarmerListings || currentFarmerListings.length === 0) {
        fullTbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No active listings found.</td></tr>';
        if (paginationEl) paginationEl.innerHTML = '';
        return;
    }

    const totalCount = currentFarmerListings.length;
    const totalPages = Math.ceil(totalCount / FARMER_PAGE_SIZE) || 1;
    if (farmerCurrentPage > totalPages) farmerCurrentPage = totalPages;
    if (farmerCurrentPage < 1) farmerCurrentPage = 1;

    const startIdx = (farmerCurrentPage - 1) * FARMER_PAGE_SIZE;
    const endIdx = startIdx + FARMER_PAGE_SIZE;
    const pagedListings = currentFarmerListings.slice(startIdx, endIdx);

    fullTbody.innerHTML = pagedListings.map(l => `
        <tr>
            <td><img src="${l.image_url}" class="table-thumb" alt="Thumb"></td>
            <td><strong>${l.species}</strong></td>
            <td>${l.breed}</td>
            <td>${l.gender}</td>
            <td>${l.age_years} Yrs</td>
            <td>${l.milk_production} L/day</td>
            <td>${l.village ? l.village + ', ' : ''}${l.district}</td>
            <td><strong>₹${l.asking_price.toLocaleString('en-IN')}</strong></td>
            <td><span class="badge-status ${l.status === 'Available' ? 'status-available' : 'status-sold'}">${formatDisplayStatus(l.status, 'farmer')}</span></td>
            <td>
                ${l.status === 'Available' ? `
                    <button class="btn btn-secondary btn-sm" onclick="openFarmerEditModal(${l.id}, ${l.asking_price}, ${l.milk_production}, '${escapeQuotes(l.address)}', '${escapeQuotes(l.pincode)}')"><i class="fa-solid fa-pen"></i> Edit</button>
                    <button class="btn btn-danger btn-sm" onclick="handleFarmerDeleteListing(${l.id})"><i class="fa-solid fa-trash"></i></button>
                ` : `<small class="text-muted">No actions (Confirmed)</small>`}
            </td>
        </tr>
    `).join('');

    renderPaginationControls('farmer-listings-pagination', farmerCurrentPage, totalPages, totalCount, 'changeFarmerPage');
}

function changeFarmerPage(newPage) {
    const totalPages = Math.ceil(currentFarmerListings.length / FARMER_PAGE_SIZE) || 1;
    if (newPage < 1 || newPage > totalPages) return;
    farmerCurrentPage = newPage;
    renderFarmerListingsTable();
}

async function loadFarmerHistory() {
    if (!currentUser) return;
    try {
        const res = await fetch(`/api/farmer/${currentUser.id}/history`);
        const history = await res.json();
        const tbody = document.getElementById('farmer-history-tbody');
        if (!tbody) return;

        if (history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="13" class="text-center text-muted">No listing history found.</td></tr>';
            return;
        }

        tbody.innerHTML = history.map(item => `
            <tr>
                <td><img src="${item.image_url}" class="table-thumb" alt="Thumb"></td>
                <td><strong>${item.species}</strong></td>
                <td>${item.breed}</td>
                <td>${item.gender}</td>
                <td>${item.age_years} Yrs</td>
                <td>${item.milk_production} L/day</td>
                <td>${item.village ? item.village + ', ' : ''}${item.district}</td>
                <td><strong>₹${item.asking_price.toLocaleString('en-IN')}</strong></td>
                <td>${formatDateTime(item.created_at)}</td>
                <td>${item.purchased_at ? formatDateTime(item.purchased_at) : '--'}</td>
                <td><strong>₹${(item.total_amount || item.asking_price).toLocaleString('en-IN')}</strong></td>
                <td><span class="badge-status ${getOrderStatusBadgeClass(item.order_status || item.status)}">${item.order_status || item.status}</span></td>
                <td>
                    ${item.delivery_status ? `<span class="badge-status ${getDeliveryStatusBadgeClass(item.delivery_status)}">${item.delivery_status}</span>` : '<span class="text-muted">--</span>'}
                </td>
            </tr>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

async function loadFarmerSold() {
    if (!currentUser) return;
    try {
        const res = await fetch(`/api/farmer/${currentUser.id}/sold`);
        const soldList = await res.json();
        const tbody = document.getElementById('farmer-sold-tbody');
        if (!tbody) return;

        if (soldList.length === 0) {
            tbody.innerHTML = '<tr><td colspan="12" class="text-center text-muted">No sold livestock records yet.</td></tr>';
            return;
        }

        tbody.innerHTML = soldList.map(item => `
            <tr>
                <td><img src="${item.image_url}" class="table-thumb" alt="Thumb"></td>
                <td><strong>${item.species}</strong></td>
                <td>${item.breed}</td>
                <td>${item.buyer_name}</td>
                <td>${item.buyer_phone}</td>
                <td>${item.delivery_district}</td>
                <td>₹${item.asking_price.toLocaleString('en-IN')}</td>
                <td><strong>₹${(item.total_amount || item.asking_price).toLocaleString('en-IN')}</strong></td>
                <td>${formatDateTime(item.purchased_at || item.created_at)}</td>
                <td><span class="badge-status ${getOrderStatusBadgeClass(item.order_status || 'CONFIRMED')}">${item.order_status || 'CONFIRMED'}</span></td>
                <td>
                    ${item.delivery_status ? `<span class="badge-status ${getDeliveryStatusBadgeClass(item.delivery_status)}">${item.delivery_status}</span>` : '<span class="text-muted">--</span>'}
                </td>
                <td>
                    ${item.proof_photo_url ? `<button class="proof-photo-btn" onclick="openViewProofModal('${item.proof_photo_url}', 'Order #${item.id || ''}')"><i class="fa-solid fa-camera"></i> Proof</button>` : ''}
                    ${item.rejection_reason ? `<small class="text-danger d-block">Rejection: ${escapeQuotes(item.rejection_reason)}</small>` : ''}
                    ${item.cancellation_reason ? `<small class="text-muted d-block">Cancelled: ${escapeQuotes(item.cancellation_reason)}</small>` : ''}
                    ${item.rating ? `<div class="star-display-read mt-1">${'★'.repeat(item.rating)}${'☆'.repeat(5 - item.rating)}</div>` : ''}
                    ${!item.proof_photo_url && !item.rejection_reason && !item.cancellation_reason && !item.rating ? '<span class="text-muted">--</span>' : ''}
                </td>
            </tr>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

async function loadFarmerInquiries() {
    if (!currentUser) return;
    try {
        const res = await fetch(`/api/inquiries/${currentUser.id}?role=farmer`);
        const inquiries = await res.json();
        const container = document.getElementById('farmer-inquiry-list');
        if (!container) return;

        if (inquiries.length === 0) {
            container.innerHTML = '<div class="alert-box alert-info"><i class="fa-solid fa-circle-info"></i> No contact requests received yet.</div>';
            return;
        }

        container.innerHTML = inquiries.map(inq => `
            <div class="card-container mb-3">
                <div class="card-header-flex">
                    <h4><i class="fa-solid fa-user"></i> ${inq.buyer_name} (${inq.buyer_phone})</h4>
                    <span class="count-badge">${formatDateTime(inq.created_at)}</span>
                </div>
                <p><strong>Livestock:</strong> ${inq.species} - ${inq.breed} (Asking Price: ₹${inq.asking_price.toLocaleString('en-IN')})</p>
                <div class="quote-box mt-2 p-2 bg-light border rounded">"${escapeQuotes(inq.message)}"</div>
            </div>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

function openFarmerEditModal(id, price, milk, address, pincode) {
    document.getElementById('edit-listing-id').value = id;
    document.getElementById('edit-asking-price').value = price;
    document.getElementById('edit-milk-production').value = milk;
    document.getElementById('edit-address').value = address || '';
    document.getElementById('edit-pincode').value = pincode || '';
    document.getElementById('modal-edit-listing').style.display = 'flex';
}

function closeEditModal() {
    document.getElementById('modal-edit-listing').style.display = 'none';
}

async function handleSaveEditedListing(e) {
    e.preventDefault();
    const id = document.getElementById('edit-listing-id').value;
    const price = document.getElementById('edit-asking-price').value;
    const milk = document.getElementById('edit-milk-production').value;
    const address = document.getElementById('edit-address').value.trim();
    const pincode = document.getElementById('edit-pincode').value.trim();

    try {
        const res = await fetch(`/api/listings/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                farmer_id: currentUser.id,
                asking_price: parseInt(price),
                milk_production: parseFloat(milk),
                address,
                pincode
            })
        });
        const data = await res.json();
        if (!res.ok) {
            showToast(data.error || 'Update failed', 'error');
            return;
        }
        showToast('Listing updated successfully!', 'success');
        closeEditModal();
        loadFarmerListings();
    } catch (err) {
        showToast('Server error.', 'error');
    }
}

async function handleFarmerDeleteListing(id) {
    if (!confirm('Are you sure you want to delete this listing?')) return;
    try {
        const res = await fetch(`/api/listings/${id}?farmer_id=${currentUser.id}`, {
            method: 'DELETE'
        });
        const data = await res.json();
        if (!res.ok) {
            showToast(data.error || 'Delete failed', 'error');
            return;
        }
        showToast('Listing deleted successfully.', 'success');
        loadFarmerListings();
    } catch (err) {
        showToast('Server error.', 'error');
    }
}

// =========================================================
// 6. BUYER DASHBOARD, MARKETPLACE & BUY NOW (COD)
// =========================================================
function switchBuyerTab(tabName) {
    document.querySelectorAll('.buyer-subtab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.dash-sidebar .side-nav-btn').forEach(b => b.classList.remove('active'));

    const tabEl = document.getElementById(`btab-${tabName}`);
    if (tabEl) tabEl.classList.add('active');

    const navBtn = document.getElementById(`bnav-${tabName}`);
    if (navBtn) navBtn.classList.add('active');

    if (tabName === 'browse') {
        loadBuyerMarketplaceData();
    } else if (tabName === 'history') {
        loadBuyerPurchaseHistory();
    } else if (tabName === 'purchased') {
        loadBuyerPurchasedLivestock();
    } else if (tabName === 'inquiries') {
        loadBuyerInquiries();
    } else if (tabName === 'profile') {
        renderBuyerProfile();
    }
}

function loadBuyerMarketplaceData() {
    if (!currentUser) return;
    document.getElementById('buyer-dash-username').textContent = currentUser.name;
    document.getElementById('buyer-dash-userloc').textContent = `${currentUser.district || 'Bengaluru'}, Karnataka`;
    fetchAvailableListings();
}

function renderBuyerProfile() {
    if (!currentUser) return;
    document.getElementById('bprof-name').textContent = currentUser.name;
    document.getElementById('bprof-email').textContent = currentUser.email;
    document.getElementById('bprof-phone').textContent = currentUser.phone;
    document.getElementById('bprof-district').textContent = currentUser.district || 'Bengaluru Urban';
    document.getElementById('bprof-taluk').textContent = currentUser.taluk || 'Yelahanka';
    document.getElementById('bprof-village').textContent = currentUser.village || 'Jala';
    document.getElementById('bprof-address').textContent = currentUser.address || '--';
    document.getElementById('bprof-pincode').textContent = currentUser.pincode || '--';
}

function populateBuyerBreedDropdown() {
    const breedSelect = document.getElementById('bfilter-breed');
    if (!breedSelect) return;
    breedSelect.innerHTML = '<option value="All">All Breeds</option>';
    const all = [...COW_BREEDS, ...BUFFALO_BREEDS].sort();
    all.forEach(b => {
        const opt = document.createElement('option');
        opt.value = b;
        opt.textContent = b;
        breedSelect.appendChild(opt);
    });
}

function onBuyerSpeciesFilterChange() {
    const species = document.getElementById('bfilter-species').value;
    const breedSelect = document.getElementById('bfilter-breed');
    breedSelect.innerHTML = '<option value="All">All Breeds</option>';

    let list = [];
    if (species === 'Cow') list = COW_BREEDS;
    else if (species === 'Buffalo') list = BUFFALO_BREEDS;
    else list = [...COW_BREEDS, ...BUFFALO_BREEDS].sort();

    list.forEach(b => {
        const opt = document.createElement('option');
        opt.value = b;
        opt.textContent = b;
        breedSelect.appendChild(opt);
    });

    applyBuyerFilters();
}

async function fetchAvailableListings() {
    try {
        const res = await fetch('/api/listings?status=Available');
        allBuyerListings = await res.json();
        applyBuyerFilters();
    } catch (e) {
        console.error(e);
    }
}

function applyBuyerFilters() {
    const search = (document.getElementById('bfilter-search').value || '').toLowerCase().trim();
    const species = document.getElementById('bfilter-species').value;
    const breed = document.getElementById('bfilter-breed').value;
    const district = document.getElementById('bfilter-district').value;
    const minMilk = parseFloat(document.getElementById('bfilter-min-milk').value) || 0;

    const minPriceInput = document.getElementById('bfilter-min-price');
    const maxPriceInput = document.getElementById('bfilter-max-price');
    const minPriceVal = minPriceInput ? minPriceInput.value.trim() : '';
    const maxPriceVal = maxPriceInput ? maxPriceInput.value.trim() : '';
    const minPrice = minPriceVal !== '' ? parseFloat(minPriceVal) : 0;
    const maxPrice = maxPriceVal !== '' ? parseFloat(maxPriceVal) : Infinity;

    const filtered = allBuyerListings.filter(item => {
        if (item.status !== 'Available') return false;
        if (species !== 'All' && item.species !== species) return false;
        if (breed !== 'All' && item.breed !== breed) return false;
        if (district && district !== 'All' && item.district !== district) return false;
        if (item.milk_production < minMilk) return false;
        
        const priceNum = Number(item.asking_price);
        if (!isNaN(minPrice) && priceNum < minPrice) return false;
        if (!isNaN(maxPrice) && priceNum > maxPrice) return false;

        if (search) {
            const matchesSearch = item.breed.toLowerCase().includes(search) ||
                                  item.species.toLowerCase().includes(search) ||
                                  item.district.toLowerCase().includes(search) ||
                                  (item.farmer_name && item.farmer_name.toLowerCase().includes(search));
            if (!matchesSearch) return false;
        }

        return true;
    });

    // Check if price filter is applied
    const isPriceFilterApplied = (minPriceVal !== '' && !isNaN(minPrice) && minPrice > 0) || 
                                 (maxPriceVal !== '' && !isNaN(maxPrice) && maxPrice < Infinity);

    // If price filter applied, sort matching records strictly in ascending numeric price order
    if (isPriceFilterApplied) {
        filtered.sort((a, b) => {
            const priceA = typeof a.asking_price === 'number' ? a.asking_price : parseFloat(a.asking_price);
            const priceB = typeof b.asking_price === 'number' ? b.asking_price : parseFloat(b.asking_price);
            return priceA - priceB;
        });
    }

    currentFilteredBuyerListings = filtered;
    buyerCurrentPage = 1;
    renderBuyerMarketplacePage();
}

function renderBuyerMarketplacePage() {
    const grid = document.getElementById('buyer-marketplace-grid');
    const countBadge = document.getElementById('buyer-listing-count');
    const paginationEl = document.getElementById('buyer-marketplace-pagination');

    const totalCount = currentFilteredBuyerListings.length;
    if (countBadge) countBadge.textContent = `Showing ${totalCount} available listing${totalCount === 1 ? '' : 's'}`;

    if (!grid) return;

    if (totalCount === 0) {
        grid.innerHTML = `
            <div class="empty-marketplace-alert-card" style="grid-column: 1/-1; text-align:center; padding: 3rem 1.5rem; background: #ffffff; border: 2px dashed #cbd5e1; border-radius: 12px; margin: 1rem 0;">
                <div style="font-size: 2.5rem; color: #94a3b8; margin-bottom: 0.75rem;"><i class="fa-solid fa-magnifying-glass"></i></div>
                <h3 style="font-size: 1.25rem; color: #1e293b; margin-bottom: 0.5rem; font-weight: 700;">No livestock matching your search is currently available.</h3>
                <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 1.25rem;">Notify me when matching livestock becomes available</p>
                <button type="button" class="btn btn-primary" id="btn-subscribe-search-alert" onclick="subscribeCurrentSearchAlert()">
                    <i class="fa-solid fa-bell"></i> Notify Me
                </button>
            </div>
        `;
        if (paginationEl) paginationEl.innerHTML = '';
        return;
    }

    const totalPages = Math.ceil(totalCount / BUYER_PAGE_SIZE) || 1;
    if (buyerCurrentPage > totalPages) buyerCurrentPage = totalPages;
    if (buyerCurrentPage < 1) buyerCurrentPage = 1;

    const startIdx = (buyerCurrentPage - 1) * BUYER_PAGE_SIZE;
    const endIdx = startIdx + BUYER_PAGE_SIZE;
    const pagedItems = currentFilteredBuyerListings.slice(startIdx, endIdx);

    renderBuyerMarketplaceGrid(pagedItems);
    renderPaginationControls('buyer-marketplace-pagination', buyerCurrentPage, totalPages, totalCount, 'changeBuyerPage');
}

function changeBuyerPage(newPage) {
    const totalPages = Math.ceil(currentFilteredBuyerListings.length / BUYER_PAGE_SIZE) || 1;
    if (newPage < 1 || newPage > totalPages) return;
    buyerCurrentPage = newPage;
    renderBuyerMarketplacePage();
}

function formatPublishedDate(dateStr) {
    if (!dateStr) return '';
    try {
        const clean = String(dateStr).trim();
        const match = clean.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
        if (match) {
            const year = match[1];
            const monthIdx = parseInt(match[2], 10) - 1;
            const day = String(parseInt(match[3], 10)).padStart(2, '0');
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            if (monthIdx >= 0 && monthIdx < 12) {
                return `${day} ${months[monthIdx]} ${year}`;
            }
        }
        const d = new Date(clean.replace(' ', 'T'));
        if (!isNaN(d.getTime())) {
            const day = String(d.getDate()).padStart(2, '0');
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            return `${day} ${months[d.getMonth()]} ${d.getFullYear()}`;
        }
    } catch (e) {
        console.warn('[Format Date Error]', e);
    }
    return '';
}

function renderBuyerMarketplaceGrid(items) {
    const grid = document.getElementById('buyer-marketplace-grid');
    if (!grid) return;

    grid.innerHTML = items.map(l => {
        const pubDate = formatPublishedDate(l.created_at);
        const pubDateHTML = pubDate ? `
            <div class="listing-published-date">
                <i class="fa-regular fa-calendar-check"></i> Published on: <strong>${pubDate}</strong>
            </div>
        ` : '';

        return `
        <div class="listing-card">
            <div class="listing-card-img-wrap">
                <img src="${l.image_url}" alt="${l.breed}">
                <span class="card-species-badge">${l.species}</span>
                <span class="card-status-tag card-status-available">AVAILABLE</span>
            </div>
            <div class="listing-card-body">
                <h4>${l.breed}</h4>
                ${pubDateHTML}
                <div class="listing-meta-row">
                    <span class="meta-pill"><i class="fa-solid fa-venus-mars"></i> ${l.gender}</span>
                    <span class="meta-pill"><i class="fa-solid fa-calendar"></i> ${l.age_years} Yrs</span>
                    ${(l.lactation_number && l.lactation_number !== 'Not Applicable') ? `<span class="meta-pill"><i class="fa-solid fa-repeat"></i> ${l.lactation_number} Lactation</span>` : ''}
                    ${(l.calf_number && l.calf_number !== 'Not Applicable') ? `<span class="meta-pill"><i class="fa-solid fa-cow"></i> ${l.calf_number} Calf</span>` : ''}
                    <span class="meta-pill"><i class="fa-solid fa-bottle-water"></i> ${l.milk_production} L/day</span>
                    <span class="meta-pill"><i class="fa-solid fa-heart-pulse"></i> ${l.health_text || 'Good'}</span>
                </div>
                <div class="listing-location-text listing-location-clickable" onclick="openLivestockLocationMap(${l.id})" title="Click to view real map & directions" role="button" tabindex="0">
                    <div style="display:flex; align-items:center; gap:0.35rem;">
                        <i class="fa-solid fa-location-dot" style="color:var(--primary);"></i>
                        <span>${l.village ? l.village + ', ' : ''}${l.taluk ? l.taluk + ', ' : ''}${l.district}</span>
                    </div>
                    <span class="location-map-indicator"><i class="fa-solid fa-map"></i> View Map</span>
                </div>
                <div class="listing-price-row">
                    <div class="price-box">
                        <span class="lbl">Livestock Price</span>
                        <span class="amt">₹${l.asking_price.toLocaleString('en-IN')}</span>
                    </div>
                    <button class="btn btn-primary btn-sm" onclick="openBuyModal(${l.id})">
                        <i class="fa-solid fa-cart-shopping"></i> BUY NOW
                    </button>
                </div>
                <div class="mt-2 text-center">
                    <button class="btn btn-link btn-sm" onclick="openContactModal(${l.id}, '${escapeQuotes(l.farmer_name)}')">
                        <i class="fa-solid fa-comments"></i> Contact Farmer (${l.farmer_name})
                    </button>
                </div>
            </div>
        </div>
        `;
    }).join('');
}

// --- Buyer Livestock Location Map & Directions Handlers ---
function openLivestockLocationMap(listingId) {
    const listing = allBuyerListings.find(l => l.id === listingId);
    if (!listing) return;

    const modal = document.getElementById('modal-livestock-map');
    const body = document.getElementById('livestock-map-modal-body');
    if (!modal || !body) return;

    // Resolve location parts from stored livestock record
    const locParts = [listing.village, listing.taluk, listing.district].filter(Boolean);
    const locationDisplay = locParts.length > 0 ? locParts.join(', ') : 'Karnataka, India';
    const fullSearchQuery = [...locParts, 'Karnataka', 'India'].join(', ');

    // Primary coordinates: GPS lat/lng if available on livestock record or resolved from DB
    const hasCoordinates = (listing.latitude != null && listing.longitude != null && !isNaN(listing.latitude) && !isNaN(listing.longitude));
    const primaryTarget = hasCoordinates 
        ? `${listing.latitude},${listing.longitude}` 
        : encodeURIComponent(fullSearchQuery);

    const embedUrl = `https://maps.google.com/maps?q=${primaryTarget}&t=&z=13&ie=UTF8&iwloc=&output=embed`;
    const directionsUrl = `https://www.google.com/maps/dir/?api=1&destination=${primaryTarget}`;
    const searchUrl = `https://www.google.com/maps/search/?api=1&query=${primaryTarget}`;

    body.innerHTML = `
        <div class="livestock-map-info-card" style="display:flex; gap:1rem; align-items:center; background:#f8fafc; padding:1rem; border-radius:8px; border:1px solid #e2e8f0; margin-bottom:1rem;">
            <img src="${listing.image_url}" alt="${listing.breed}" style="width:75px; height:75px; object-fit:cover; border-radius:8px; border:1px solid #cbd5e1;">
            <div style="flex:1;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                    <h4 style="margin:0; font-size:1.05rem; color:#0f172a;">${listing.species} &bull; ${listing.breed}</h4>
                    <span style="font-weight:700; color:#166534; font-size:1rem;">₹${listing.asking_price.toLocaleString('en-IN')}</span>
                </div>
                <div style="font-size:0.85rem; color:#475569; margin-bottom:0.25rem;">
                    <i class="fa-solid fa-location-dot" style="color:#ef4444;"></i> Stored Location: <strong>${locationDisplay}</strong>
                </div>
                <div style="font-size:0.8rem; color:#64748b; display:flex; gap:1rem; flex-wrap:wrap;">
                    <span>Farmer: <strong>${escapeQuotes(listing.farmer_name)}</strong></span>
                    ${hasCoordinates ? `<span class="badge" style="background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:4px; font-weight:600;"><i class="fa-solid fa-crosshairs"></i> GPS: ${Number(listing.latitude).toFixed(4)}°, ${Number(listing.longitude).toFixed(4)}°</span>` : `<span class="badge" style="background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:4px;"><i class="fa-solid fa-landmark"></i> Administrative Region</span>`}
                </div>
            </div>
        </div>

        <div class="map-embed-card">
            <iframe id="livestock-map-iframe" src="${embedUrl}" loading="lazy" allowfullscreen referrerpolicy="no-referrer-when-downgrade" title="Livestock Location Map"></iframe>
        </div>

        <div class="alert-box alert-info mb-3" style="font-size:0.85rem;">
            <i class="fa-solid fa-route"></i>
            <div>Click <strong>Get Directions</strong> to navigate from your current location directly to this livestock in ${locationDisplay}.</div>
        </div>

        <div class="map-action-grid">
            <a href="${directionsUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-primary" style="text-decoration:none; display:flex; align-items:center; justify-content:center; gap:0.5rem; padding:0.75rem 1rem;">
                <i class="fa-solid fa-diamond-turn-right"></i> Get Directions (from My Location)
            </a>
            <button type="button" class="btn btn-secondary" onclick="navigateFromBuyerLocation('${primaryTarget}')" style="display:flex; align-items:center; justify-content:center; gap:0.5rem; padding:0.75rem 1rem;">
                <i class="fa-solid fa-location-crosshairs"></i> Navigate with Live Device GPS
            </button>
        </div>
        <div style="text-align:center; margin-top:0.75rem;">
            <a href="${searchUrl}" target="_blank" rel="noopener noreferrer" style="font-size:0.825rem; color:#2563eb; text-decoration:none;">
                <i class="fa-solid fa-arrow-up-right-from-square"></i> Open in Google Maps full app
            </a>
        </div>
    `;

    modal.style.display = 'flex';
}

function closeLivestockMapModal() {
    const modal = document.getElementById('modal-livestock-map');
    if (modal) modal.style.display = 'none';
    const iframe = document.getElementById('livestock-map-iframe');
    if (iframe) iframe.src = 'about:blank';
}

function navigateFromBuyerLocation(destination) {
    if (!navigator.geolocation) {
        window.open(`https://www.google.com/maps/dir/?api=1&destination=${destination}`, '_blank');
        return;
    }
    showToast('Acquiring current GPS position...', 'info');
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            const origin = `${pos.coords.latitude},${pos.coords.longitude}`;
            const url = `https://www.google.com/maps/dir/?api=1&origin=${origin}&destination=${destination}`;
            window.open(url, '_blank');
        },
        (err) => {
            console.warn('Geolocation denied or unavailable, opening directions directly:', err);
            window.open(`https://www.google.com/maps/dir/?api=1&destination=${destination}`, '_blank');
        },
        { timeout: 7000 }
    );
}

// --- Common Pagination Renderer ---
function renderPaginationControls(containerId, currentPage, totalPages, totalCount, changeFnName) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (totalPages <= 1) {
        container.innerHTML = totalCount > 0 
            ? `<div class="pagination-info" style="border:none; margin:0; text-align:center;">Showing all ${totalCount} record${totalCount === 1 ? '' : 's'} (Page 1 of 1)</div>`
            : '';
        return;
    }

    let pagesHtml = '';
    for (let p = 1; p <= totalPages; p++) {
        const isActive = p === currentPage ? 'active' : '';
        pagesHtml += `<button type="button" class="pagination-page-btn ${isActive}" onclick="${changeFnName}(${p})">${p}</button>`;
    }

    container.innerHTML = `
        <div class="pagination-container">
            <button type="button" class="pagination-btn" ${currentPage <= 1 ? 'disabled' : ''} onclick="${changeFnName}(${currentPage - 1})">
                <i class="fa-solid fa-chevron-left"></i> Previous
            </button>
            <div class="pagination-pages">
                ${pagesHtml}
            </div>
            <button type="button" class="pagination-btn" ${currentPage >= totalPages ? 'disabled' : ''} onclick="${changeFnName}(${currentPage + 1})">
                Next <i class="fa-solid fa-chevron-right"></i>
            </button>
            <span class="pagination-info">Page ${currentPage} of ${totalPages} (${totalCount} records)</span>
        </div>
    `;
}

// --- Buy Now Modal (Zero Delivery Charges) ---
async function openBuyModal(listingId) {
    if (!currentUser || currentUser.role !== 'buyer') {
        showToast('Please login as Buyer to purchase livestock.', 'warning');
        navigateTo('login');
        selectLoginRole('buyer');
        return;
    }

    const listing = allBuyerListings.find(l => l.id === listingId);
    if (!listing) return;
    currentBuyItem = listing;

    const modalBody = document.getElementById('buy-modal-body');
    modalBody.innerHTML = `
        <div class="alert-box alert-success mb-3">
            <i class="fa-solid fa-truck-fast"></i>
            <div><strong>Zero Delivery Charges Across Karnataka</strong>: Pay the exact livestock price upon delivery via Cash on Delivery.</div>
        </div>

        <!-- STEP 1: Livestock Details Specification -->
        <div class="purchase-livestock-preview mb-3" style="display:flex; gap:1.25rem; align-items:center; background:#f8fafc; padding:1rem; border-radius:8px; border:1px solid #e2e8f0;">
            <img src="${listing.image_url}" style="width:85px; height:85px; object-fit:cover; border-radius:8px;" alt="${listing.breed}">
            <div style="flex:1;">
                <div style="font-size:0.85rem; color:#64748b; margin-bottom:2px;">Livestock: <strong style="color:#0f172a;">${listing.species}</strong></div>
                <div style="font-size:0.85rem; color:#64748b; margin-bottom:2px;">Breed: <strong style="color:#0f172a;">${listing.breed}</strong> (${listing.gender})</div>
                <div style="font-size:0.85rem; color:#64748b; margin-bottom:2px;">Age: <strong style="color:#0f172a;">${listing.age_years} Years</strong></div>
                <div style="font-size:0.85rem; color:#64748b; margin-bottom:2px;">Milk Production: <strong style="color:#0f172a;">${listing.milk_production} L/day</strong></div>
                <div style="font-size:0.85rem; color:#64748b; margin-bottom:4px;">Seller: <strong style="color:#0f172a;">${listing.farmer_name}</strong> (${listing.farmer_phone})</div>
                <div style="font-size:0.95rem; font-weight:700; color:#166534;">Estimated Price: ₹${listing.asking_price.toLocaleString('en-IN')}</div>
            </div>
        </div>

        <div class="form-divider">Delivery Destination (Karnataka)</div>

        <div class="form-row">
            <div class="form-group">
                <label>Delivery District <span class="required">*</span></label>
                <select id="buy-delivery-district" required>
                    <!-- Populated -->
                </select>
            </div>
            <div class="form-group">
                <label>Delivery Taluk <span class="required">*</span></label>
                <input type="text" id="buy-delivery-taluk" value="${currentUser.taluk || ''}" placeholder="Enter Taluk" required>
            </div>
        </div>

        <div class="form-row">
            <div class="form-group" style="flex:2;">
                <label>Delivery Address / Landmark <span class="required">*</span></label>
                <input type="text" id="buy-delivery-address" value="${currentUser.address || ''}" required placeholder="Doorstep delivery address">
            </div>
            <div class="form-group" style="flex:1;">
                <label>Pincode <span class="required">*</span></label>
                <input type="text" id="buy-delivery-pincode" value="${currentUser.pincode || ''}" required pattern="[0-9]{6}" placeholder="PIN">
            </div>
        </div>

        <!-- Live Purchase Summary Card (Zero Delivery Fee) -->
        <div class="purchase-summary-card mb-3" id="buy-summary-card">
            <div class="summary-line">
                <span>Livestock Price:</span>
                <span>₹${listing.asking_price.toLocaleString('en-IN')}</span>
            </div>
            <div class="summary-line">
                <span>Delivery Charges:</span>
                <span style="color:#16a34a; font-weight:700;">₹0 (FREE Delivery)</span>
            </div>
            <div class="summary-line summary-total">
                <span>TOTAL AMOUNT (COD):</span>
                <span>₹${listing.asking_price.toLocaleString('en-IN')}</span>
            </div>
        </div>

        <!-- Action: Direct Purchase Confirmation (NO PURCHASE OTP) -->
        <div id="buy-action-box">
            <button type="button" class="btn btn-primary btn-block btn-lg" id="btn-confirm-purchase" onclick="confirmDirectPurchase(${listing.id})">
                <i class="fa-solid fa-circle-check"></i> Confirm Purchase
            </button>
        </div>
    `;

    populateDistrictSelect('buy-delivery-district', 'Select Delivery District');
    document.getElementById('buy-delivery-district').value = currentUser.district || 'Bengaluru Urban';
    document.getElementById('modal-buy-now').style.display = 'flex';
}

function closeBuyModal() {
    document.getElementById('modal-buy-now').style.display = 'none';
    currentBuyItem = null;
}

async function confirmDirectPurchase(listingId) {
    const deliveryDistrict = document.getElementById('buy-delivery-district').value;
    const deliveryTaluk = document.getElementById('buy-delivery-taluk').value.trim();
    const deliveryAddress = document.getElementById('buy-delivery-address').value.trim();
    const deliveryPincode = document.getElementById('buy-delivery-pincode').value.trim();

    if (!deliveryDistrict || !deliveryAddress) {
        showToast('Please provide your complete delivery address before confirming purchase.', 'warning');
        return;
    }

    const btn = document.getElementById('btn-confirm-purchase');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Purchase...';

    const payload = {
        buyer_id: currentUser.id,
        buyer_name: currentUser.name,
        buyer_phone: currentUser.phone,
        delivery_district: deliveryDistrict,
        delivery_taluk: deliveryTaluk,
        delivery_address: deliveryAddress,
        delivery_pincode: deliveryPincode
    };

    try {
        const res = await fetch(`/api/listings/${listingId}/purchase`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-circle-check"></i> Confirm Purchase';

        if (!res.ok) {
            showToast(data.error || 'Failed to complete purchase.', 'error');
            return;
        }

        showToast(data.message || 'Purchase confirmed successfully.', 'success');
        closeBuyModal();
        switchBuyerTab('purchased');
        fetchAvailableListings();

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-circle-check"></i> Confirm Purchase';
        showToast('Server error during purchase confirmation.', 'error');
    }
}

async function subscribeCurrentSearchAlert() {
    const species = document.getElementById('bfilter-species') ? document.getElementById('bfilter-species').value : '';
    const breed = document.getElementById('bfilter-breed') ? document.getElementById('bfilter-breed').value : '';
    const district = document.getElementById('bfilter-district') ? document.getElementById('bfilter-district').value : '';
    const minMilk = document.getElementById('bfilter-min-milk') ? document.getElementById('bfilter-min-milk').value : '';
    const maxPrice = document.getElementById('bfilter-max-price') ? document.getElementById('bfilter-max-price').value : '';

    let buyerEmail = '';
    let buyerId = null;

    if (currentUser && currentUser.role === 'buyer') {
        buyerEmail = currentUser.email;
        buyerId = currentUser.id;
    } else {
        buyerEmail = prompt('Please enter your email to receive livestock availability alerts:');
        if (!buyerEmail || !buyerEmail.trim()) return;
        buyerEmail = buyerEmail.trim().toLowerCase();
    }

    const btn = document.getElementById('btn-subscribe-search-alert');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving Alert...';
    }

    try {
        const res = await fetch('/api/alerts/subscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                buyer_id: buyerId,
                buyer_email: buyerEmail,
                species: species !== 'All' ? species : '',
                breed: breed !== 'All' ? breed : '',
                district: district !== 'All' ? district : '',
                minimum_milk_production: minMilk ? parseFloat(minMilk) : null,
                maximum_price: maxPrice ? parseInt(maxPrice) : null
            })
        });
        const data = await res.json();

        if (btn) {
            btn.innerHTML = '<i class="fa-solid fa-check"></i> Subscribed!';
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-success');
        }

        if (!res.ok) {
            showToast(data.error || 'Failed to save alert.', 'error');
            if (btn) btn.disabled = false;
            return;
        }

        showToast(data.message || 'Availability alert saved! We will notify you by email when matching livestock becomes available.', 'success');

    } catch (err) {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-bell"></i> Notify Me';
        }
        showToast('Failed to connect to server.', 'error');
    }
}

// --- Buyer Purchase History & Purchased Loaders ---
async function loadBuyerPurchaseHistory() {
    if (!currentUser) return;
    try {
        const res = await fetch(`/api/buyer/${currentUser.id}/purchases`);
        const purchases = await res.json();
        const tbody = document.getElementById('buyer-history-tbody');
        if (!tbody) return;

        if (purchases.length === 0) {
            tbody.innerHTML = '<tr><td colspan="14" class="text-center text-muted">No purchase history found. Browse available livestock to place an order.</td></tr>';
            return;
        }

        tbody.innerHTML = purchases.map(p => `
            <tr>
                <td><img src="${p.image_url || '/static/uploads/default.jpg'}" class="table-thumb" alt="Thumb"></td>
                <td><strong>${p.species || 'Livestock'}</strong></td>
                <td>${p.breed || 'Standard'}</td>
                <td>${p.gender || 'Female'}</td>
                <td>${p.farmer_name} (${p.farmer_phone})</td>
                <td>${p.farmer_village ? p.farmer_village + ', ' : ''}${p.farmer_district || 'Karnataka'}</td>
                <td>${p.delivery_address || p.delivery_district}</td>
                <td>${formatDateTime(p.purchased_at || p.created_at)}</td>
                <td>₹${p.livestock_price.toLocaleString('en-IN')}</td>
                <td><strong>₹${(p.total_amount || p.livestock_price).toLocaleString('en-IN')}</strong></td>
                <td><span class="cod-badge">CASH ON DELIVERY</span></td>
                <td><span class="badge-status ${getOrderStatusBadgeClass(p.order_status || 'CONFIRMED')}">${p.order_status || 'CONFIRMED'}</span></td>
                <td>
                    ${p.delivery_status ? `<span class="badge-status ${getDeliveryStatusBadgeClass(p.delivery_status)}">${p.delivery_status}</span>` : '<span class="text-muted">--</span>'}
                </td>
                <td>
                    ${p.rating ? `<div class="star-display-read">${'★'.repeat(p.rating)}${'☆'.repeat(5 - p.rating)}</div>` : ''}
                    ${p.proof_photo_url ? `<button class="proof-photo-btn mt-1" onclick="openViewProofModal('${p.proof_photo_url}', 'Order #${p.id}')"><i class="fa-solid fa-camera"></i> Proof</button>` : ''}
                    ${!p.rating && !p.proof_photo_url ? '<span class="text-muted">--</span>' : ''}
                </td>
            </tr>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

async function loadBuyerPurchasedLivestock() {
    if (!currentUser) return;
    try {
        const res = await fetch(`/api/buyer/${currentUser.id}/purchases`);
        const purchases = await res.json();
        const grid = document.getElementById('buyer-purchased-grid');
        if (!grid) return;

        if (purchases.length === 0) {
            grid.innerHTML = '<div class="alert-box alert-info" style="grid-column:1/-1;"><i class="fa-solid fa-circle-info"></i> You have not purchased any livestock yet. Browse available listings to place an order.</div>';
            return;
        }

        grid.innerHTML = purchases.map(p => {
            const orderStatus = p.order_status || 'CONFIRMED';
            const deliveryStatus = p.delivery_status || '';

            let actionHtml = '';
            if (orderStatus === 'CONFIRMED') {
                actionHtml = `
                    <div class="order-actions-wrap">
                        <button class="btn btn-primary btn-sm" onclick="openMarkPurchasedModal(${p.id})">
                            <i class="fa-solid fa-check"></i> Mark as Purchased
                        </button>
                        <button class="btn btn-secondary btn-sm" onclick="openRejectDeliveryModal(${p.id})">
                            <i class="fa-solid fa-xmark"></i> Reject Delivery
                        </button>
                        <button class="btn btn-danger btn-sm" onclick="openCancelOrderModal(${p.id})">
                            <i class="fa-solid fa-ban"></i> Cancel Order
                        </button>
                    </div>
                `;
            } else if (orderStatus === 'PURCHASED') {
                actionHtml = `
                    <div class="order-actions-wrap">
                        ${p.proof_photo_url ? `<button class="proof-photo-btn" onclick="openViewProofModal('${p.proof_photo_url}', 'Order #${p.id}')"><i class="fa-solid fa-camera"></i> View Delivery Proof</button>` : ''}
                        ${p.rating ? `
                            <div class="mt-2">
                                <span class="star-display-read">${'★'.repeat(p.rating)}${'☆'.repeat(5 - p.rating)}</span>
                                ${p.review ? `<p class="text-muted mt-1" style="font-size:0.8rem;">"${escapeQuotes(p.review)}"</p>` : ''}
                            </div>
                        ` : `
                            <button class="btn btn-primary btn-sm" onclick="openReviewModal(${p.id})">
                                <i class="fa-solid fa-star"></i> Rate & Review
                            </button>
                        `}
                    </div>
                `;
            } else if (deliveryStatus === 'REJECTED') {
                actionHtml = `
                    <div class="order-actions-wrap">
                        <div class="alert-box alert-danger p-2 mb-2" style="font-size:0.8rem;">
                            <strong>Delivery Rejected:</strong> ${escapeQuotes(p.rejection_reason || 'Rejected by buyer')}
                        </div>
                        ${p.rating ? `
                            <div class="star-display-read">${'★'.repeat(p.rating)}${'☆'.repeat(5 - p.rating)}</div>
                        ` : `
                            <button class="btn btn-secondary btn-sm" onclick="openReviewModal(${p.id})">
                                <i class="fa-solid fa-star"></i> Rate & Review
                            </button>
                        `}
                    </div>
                `;
            } else if (orderStatus === 'CANCELLED') {
                actionHtml = `
                    <div class="alert-box alert-warning p-2 mt-2" style="font-size:0.8rem;">
                        <strong>Order Cancelled:</strong> ${escapeQuotes(p.cancellation_reason || 'Buyer cancelled')}
                        ${p.cancellation_details ? ` - ${escapeQuotes(p.cancellation_details)}` : ''}
                    </div>
                `;
            }

            return `
                <div class="listing-card">
                    <div class="listing-card-img-wrap">
                        <img src="${p.image_url || '/static/uploads/default.jpg'}" alt="${p.breed}">
                        <span class="card-species-badge">${p.species}</span>
                        <span class="card-status-tag ${getOrderStatusBadgeClass(orderStatus)}">${orderStatus}</span>
                    </div>
                    <div class="listing-card-body">
                        <h4>${p.breed}</h4>
                        <div class="listing-meta-row">
                            <span class="meta-pill"><i class="fa-solid fa-user"></i> Seller: ${p.farmer_name}</span>
                            <span class="meta-pill"><i class="fa-solid fa-phone"></i> ${p.farmer_phone}</span>
                            ${deliveryStatus ? `<span class="badge-status ${getDeliveryStatusBadgeClass(deliveryStatus)}">${deliveryStatus}</span>` : ''}
                        </div>
                        <div class="purchase-breakdown-box p-2 bg-light border rounded mt-2" style="font-size:0.85rem;">
                            <div>Livestock Price: <strong>₹${p.livestock_price.toLocaleString('en-IN')}</strong></div>
                            <div>Delivery Fee: <strong class="text-success">₹0 (FREE Delivery)</strong></div>
                            <div style="font-weight:700; color:#2e7d32; border-top:1px solid #e2e8f0; margin-top:4px; padding-top:4px;">
                                Total (Cash on Delivery): ₹${(p.total_amount || p.livestock_price).toLocaleString('en-IN')}
                            </div>
                        </div>
                        <div class="mt-2 text-muted" style="font-size:0.75rem;">
                            <i class="fa-solid fa-clock"></i> Placed on ${formatDateTime(p.purchased_at || p.created_at)}
                        </div>
                        ${actionHtml}
                    </div>
                </div>
            `;
        }).join('');
    } catch (e) {
        console.error(e);
    }
}

async function loadBuyerInquiries() {
    if (!currentUser) return;
    try {
        const res = await fetch(`/api/inquiries/${currentUser.id}?role=buyer`);
        const inquiries = await res.json();
        const container = document.getElementById('buyer-inquiry-list');
        if (!container) return;

        if (inquiries.length === 0) {
            container.innerHTML = '<div class="alert-box alert-info"><i class="fa-solid fa-circle-info"></i> You have not sent any inquiries.</div>';
            return;
        }

        container.innerHTML = inquiries.map(inq => `
            <div class="card-container mb-3">
                <div class="card-header-flex">
                    <h4><i class="fa-solid fa-cow"></i> ${inq.species} - ${inq.breed}</h4>
                    <span class="count-badge">${formatDateTime(inq.created_at)}</span>
                </div>
                <p><strong>Farmer:</strong> ${inq.farmer_name} (${inq.farmer_phone}) | Asking Price: ₹${inq.asking_price.toLocaleString('en-IN')}</p>
                <div class="quote-box mt-2 p-2 bg-light border rounded">"${escapeQuotes(inq.message)}"</div>
            </div>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

function openContactModal(listingId, farmerName) {
    document.getElementById('inquiry-listing-id').value = listingId;
    document.getElementById('inquiry-farmer-name').value = farmerName;
    document.getElementById('inquiry-message').value = '';
    document.getElementById('modal-contact-farmer').style.display = 'flex';
}

function closeContactModal() {
    document.getElementById('modal-contact-farmer').style.display = 'none';
}

async function handleSendInquiry(e) {
    e.preventDefault();
    if (!currentUser) {
        showToast('Please login to send messages.', 'warning');
        return;
    }
    const listingId = document.getElementById('inquiry-listing-id').value;
    const message = document.getElementById('inquiry-message').value.trim();

    try {
        const res = await fetch(`/api/listings/${listingId}/contact`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                buyer_id: currentUser.id,
                buyer_name: currentUser.name,
                buyer_phone: currentUser.phone,
                message
            })
        });
        const data = await res.json();
        if (!res.ok) {
            showToast(data.error || 'Failed to send inquiry.', 'error');
            return;
        }
        showToast(data.message || 'Message sent successfully to farmer!', 'success');
        closeContactModal();
    } catch (err) {
        showToast('Server error.', 'error');
    }
}

// =========================================================
// 7. ADMIN DASHBOARD & 15-DAY SOLD-OUT REMOVAL RULE
// =========================================================
function switchAdminTab(tabName) {
    document.querySelectorAll('.admin-subtab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.admin-sidebar .side-nav-btn').forEach(b => b.classList.remove('active'));

    const tabEl = document.getElementById(`atab-${tabName}`);
    if (tabEl) tabEl.classList.add('active');

    const navBtn = document.getElementById(`anav-${tabName}`);
    if (navBtn) navBtn.classList.add('active');

    if (tabName === 'overview') {
        loadAdminOverviewStats();
    } else if (tabName === 'farmers') {
        loadAdminFarmers();
    } else if (tabName === 'buyers') {
        loadAdminBuyers();
    } else if (tabName === 'livestock') {
        loadAdminLivestock();
    } else if (tabName === 'purchases') {
        loadAdminPurchases();
    } else if (tabName === 'soldout') {
        loadAdminSoldOutManagement();
    } else if (tabName === 'contacts') {
        loadAdminContacts();
    }
}

function loadAdminDashboardData() {
    if (!currentUser || currentUser.role !== 'admin') return;
    switchAdminTab('overview');
}

async function loadAdminOverviewStats() {
    try {
        const res = await fetch(`/api/admin/dashboard?admin_id=${currentUser.id}`);
        const data = await res.json();

        document.getElementById('ametric-farmers').textContent = data.total_farmers;
        document.getElementById('ametric-buyers').textContent = data.total_buyers;
        document.getElementById('ametric-livestock').textContent = data.total_livestock;
        document.getElementById('ametric-available').textContent = data.available_livestock;
        document.getElementById('ametric-sold').textContent = data.sold_livestock;
        document.getElementById('ametric-purchases').textContent = data.total_purchases;

        // Load recent purchases
        const purchRes = await fetch('/api/admin/purchases');
        const purchases = await purchRes.json();
        const tbody = document.getElementById('admin-recent-purchases-tbody');
        if (tbody) {
            if (purchases.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No purchases recorded yet.</td></tr>';
            } else {
                tbody.innerHTML = purchases.slice(0, 5).map(p => `
                    <tr>
                        <td>#${p.id}</td>
                        <td>${p.buyer_name}</td>
                        <td>${p.farmer_name}</td>
                        <td>${p.species} (${p.breed})</td>
                        <td><strong>₹${p.total_amount.toLocaleString('en-IN')}</strong></td>
                        <td>${formatDateTime(p.purchased_at || p.created_at)}</td>
                        <td><span class="badge-status status-sold">CONFIRMED (COD)</span></td>
                    </tr>
                `).join('');
            }
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadAdminFarmers() {
    try {
        const res = await fetch('/api/admin/farmers');
        const farmers = await res.json();
        const tbody = document.getElementById('admin-farmers-tbody');
        if (!tbody) return;

        if (farmers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="11" class="text-center text-muted">No registered farmers.</td></tr>';
            return;
        }

        tbody.innerHTML = farmers.map(f => `
            <tr>
                <td>${f.id}</td>
                <td><strong>${f.name}</strong></td>
                <td>${f.phone}</td>
                <td>${f.email}</td>
                <td>${f.district || '--'}</td>
                <td>${f.taluk || '--'}</td>
                <td>${f.village || '--'}</td>
                <td>${f.total_listings || 0}</td>
                <td>${f.sold_count || 0}</td>
                <td>${formatDateTime(f.created_at)}</td>
                <td><span class="badge-status status-available">${f.status || 'Active'}</span></td>
            </tr>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

async function loadAdminBuyers() {
    try {
        const res = await fetch('/api/admin/buyers');
        const buyers = await res.json();
        const tbody = document.getElementById('admin-buyers-tbody');
        if (!tbody) return;

        if (buyers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No registered buyers.</td></tr>';
            return;
        }

        tbody.innerHTML = buyers.map(b => `
            <tr>
                <td>${b.id}</td>
                <td><strong>${b.name}</strong></td>
                <td>${b.phone}</td>
                <td>${b.email}</td>
                <td>${b.district || '--'}</td>
                <td>${b.taluk || '--'}</td>
                <td>${b.village || '--'}</td>
                <td>${b.purchase_count || 0}</td>
                <td>${formatDateTime(b.created_at)}</td>
                <td><span class="badge-status status-available">${b.status || 'Active'}</span></td>
            </tr>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

async function loadAdminLivestock() {
    try {
        const res = await fetch('/api/admin/livestock');
        allAdminLivestock = await res.json();
        renderAdminLivestockTable(allAdminLivestock);
    } catch (e) {
        console.error(e);
    }
}

function filterAdminLivestock(filterType, btnEl) {
    document.querySelectorAll('.admin-filter-pills .filter-pill').forEach(b => b.classList.remove('active'));
    if (btnEl) btnEl.classList.add('active');

    let filtered = allAdminLivestock;
    if (filterType === 'Available') {
        filtered = allAdminLivestock.filter(l => l.status === 'Available');
    } else if (filterType === 'sold_less_15') {
        filtered = allAdminLivestock.filter(l => l.status === 'SOLD_OUT' && !l.can_remove);
    } else if (filterType === 'sold_eligible_15') {
        filtered = allAdminLivestock.filter(l => l.status === 'SOLD_OUT' && l.can_remove);
    } else if (filterType === 'REMOVED') {
        filtered = allAdminLivestock.filter(l => l.status === 'REMOVED');
    }

    renderAdminLivestockTable(filtered);
}

function renderAdminLivestockTable(items) {
    const tbody = document.getElementById('admin-livestock-tbody');
    if (!tbody) return;

    if (items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="13" class="text-center text-muted">No livestock listings found.</td></tr>';
        return;
    }

    tbody.innerHTML = items.map(l => `
        <tr>
            <td>${l.id}</td>
            <td><img src="${l.image_url}" class="table-thumb" alt="Thumb"></td>
            <td><strong>${l.species}</strong></td>
            <td>${l.breed}</td>
            <td>${l.gender}</td>
            <td>${l.age_years} Yrs</td>
            <td>${l.milk_production} L/day</td>
            <td>${l.farmer_name}</td>
            <td>${l.village ? l.village + ', ' : ''}${l.district}</td>
            <td><strong>₹${l.asking_price.toLocaleString('en-IN')}</strong></td>
            <td>${formatDateTime(l.created_at)}</td>
            <td>${l.purchased_at ? formatDateTime(l.purchased_at) : '--'}</td>
            <td>
                <span class="badge-status ${getStatusBadgeClass(l.status, l.can_remove)}">
                    ${l.status === 'SOLD_OUT' && l.can_remove ? 'ELIGIBLE FOR REMOVAL' : (l.status === 'SOLD_OUT' ? 'SOLD OUT' : l.status)}
                </span>
            </td>
        </tr>
    `).join('');
}

async function loadAdminPurchases() {
    try {
        const res = await fetch('/api/admin/purchases');
        const purchases = await res.json();
        const tbody = document.getElementById('admin-purchases-tbody');
        if (!tbody) return;

        if (purchases.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No transactions found.</td></tr>';
            return;
        }

        tbody.innerHTML = purchases.map(p => `
            <tr>
                <td>#${p.id}</td>
                <td><strong>${p.buyer_name}</strong> (${p.buyer_phone})</td>
                <td>${p.farmer_name} (${p.farmer_phone})</td>
                <td>${p.species} - ${p.breed}</td>
                <td>₹${p.livestock_price.toLocaleString('en-IN')}</td>
                <td><strong>₹${(p.total_amount || p.livestock_price).toLocaleString('en-IN')}</strong></td>
                <td>${formatDateTime(p.purchased_at || p.created_at)}</td>
                <td><span class="badge-status ${getOrderStatusBadgeClass(p.order_status || 'CONFIRMED')}">${p.order_status || 'CONFIRMED'}</span></td>
                <td>
                    ${p.delivery_status ? `<span class="badge-status ${getDeliveryStatusBadgeClass(p.delivery_status)}">${p.delivery_status}</span>` : '<span class="text-muted">--</span>'}
                </td>
                <td>
                    ${p.proof_photo_url ? `<button class="proof-photo-btn" onclick="openViewProofModal('${p.proof_photo_url}', 'Order #${p.id}')"><i class="fa-solid fa-camera"></i> Proof</button>` : ''}
                    ${p.rejection_reason ? `<small class="text-danger d-block">Rejection: ${escapeQuotes(p.rejection_reason)}</small>` : ''}
                    ${p.cancellation_reason ? `<small class="text-muted d-block">Cancelled: ${escapeQuotes(p.cancellation_reason)}</small>` : ''}
                    ${p.rating ? `<div class="star-display-read mt-1">${'★'.repeat(p.rating)}${'☆'.repeat(5 - p.rating)}</div>` : ''}
                    ${!p.proof_photo_url && !p.rejection_reason && !p.cancellation_reason && !p.rating ? '<span class="text-muted">--</span>' : ''}
                </td>
            </tr>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

async function loadAdminSoldOutManagement() {
    try {
        const res = await fetch('/api/admin/livestock');
        const livestock = await res.json();
        const soldOutList = livestock.filter(l => l.status === 'SOLD_OUT');
        const tbody = document.getElementById('admin-soldout-tbody');
        if (!tbody) return;

        if (soldOutList.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No sold-out livestock currently pending removal.</td></tr>';
            return;
        }

        tbody.innerHTML = soldOutList.map(l => `
            <tr>
                <td>#${l.id}</td>
                <td><img src="${l.image_url}" class="table-thumb" alt="Thumb"></td>
                <td><strong>${l.species}</strong> - ${l.breed}</td>
                <td>${l.farmer_name}</td>
                <td>${l.buyer_name || '--'}</td>
                <td>${l.purchased_at ? formatDateTime(l.purchased_at) : '--'}</td>
                <td>${l.days_since_purchase} day(s)</td>
                <td>${l.eligible_removal_date || '--'}</td>
                <td>
                    <span class="badge-status ${l.can_remove ? 'status-eligible' : 'status-sold'}">
                        ${l.can_remove ? 'ELIGIBLE FOR REMOVAL' : 'RETENTION (' + l.days_remaining + 'd left)'}
                    </span>
                </td>
                <td>
                    ${l.can_remove ? `
                        <button class="btn btn-danger btn-sm" onclick="openAdminRemoveModal(${l.id}, '${escapeQuotes(l.species)} - ${escapeQuotes(l.breed)}', '${l.purchased_at}', '${l.eligible_removal_date}')">
                            <i class="fa-solid fa-trash-can"></i> REMOVE LISTING
                        </button>
                    ` : `
                        <button class="btn btn-secondary btn-sm" disabled title="Removal locked until 15 days pass from purchase date">
                            <i class="fa-solid fa-lock"></i> Protected
                        </button>
                    `}
                </td>
            </tr>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

async function loadAdminContacts() {
    try {
        const res = await fetch('/api/admin/contacts');
        const contacts = await res.json();
        const tbody = document.getElementById('admin-contacts-tbody');
        if (!tbody) return;

        if (contacts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No contact inquiries.</td></tr>';
            return;
        }

        tbody.innerHTML = contacts.map(c => `
            <tr>
                <td>#${c.id}</td>
                <td>${c.buyer_name}</td>
                <td>${c.buyer_phone}</td>
                <td>${c.farmer_id}</td>
                <td>${escapeQuotes(c.message)}</td>
                <td>${formatDateTime(c.created_at)}</td>
                <td><span class="badge-status status-available">${c.status}</span></td>
            </tr>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}

// Admin 15-Day Removal Modal
function openAdminRemoveModal(id, breed, purchDate, eligDate) {
    currentAdminRemoveListing = { id, breed, purchDate, eligDate };
    document.getElementById('rem-modal-id').textContent = `#${id}`;
    document.getElementById('rem-modal-breed').textContent = breed;
    document.getElementById('rem-modal-purchdate').textContent = purchDate ? formatDateTime(purchDate) : '--';
    document.getElementById('rem-modal-eligdate').textContent = eligDate || '--';
    document.getElementById('modal-admin-remove').style.display = 'flex';
}

function closeAdminRemoveModal() {
    document.getElementById('modal-admin-remove').style.display = 'none';
    currentAdminRemoveListing = null;
}

async function executeAdminListingRemoval() {
    if (!currentAdminRemoveListing) return;
    const listingId = currentAdminRemoveListing.id;
    const btn = document.getElementById('btn-confirm-admin-remove');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Removing...';

    try {
        const res = await fetch(`/api/admin/listings/${listingId}/remove-soldout`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ admin_id: currentUser.id })
        });
        const data = await res.json();

        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-trash-can"></i> Remove Listing';

        if (!res.ok) {
            showToast(data.error || 'Removal rejected by system.', 'error');
            return;
        }

        showToast('Sold-out listing removed successfully. Purchase history preserved.', 'success');
        closeAdminRemoveModal();
        loadAdminSoldOutManagement();
        loadAdminLivestock();

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-trash-can"></i> Remove Listing';
        showToast('Server error during removal.', 'error');
    }
}

// =========================================================
// 8. UTILITIES & TOAST NOTIFICATIONS
// =========================================================
function formatDisplayStatus(status, role = 'farmer') {
    if (!status || status === 'Available') return 'AVAILABLE';
    if (status === 'SOLD_OUT') {
        if (role === 'admin') return 'SOLD OUT';
        return 'CONFIRMED';
    }
    if (status === 'REMOVED') return 'REMOVED';
    return status;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = 'fa-circle-info';
    if (type === 'success') icon = 'fa-circle-check';
    else if (type === 'error') icon = 'fa-triangle-exclamation';
    else if (type === 'warning') icon = 'fa-circle-exclamation';

    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${escapeQuotes(message)}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 250);
    }, 4000);
}

function formatDateTime(dtStr) {
    if (!dtStr) return '--';
    try {
        const d = new Date(dtStr.replace(' ', 'T'));
        if (isNaN(d.getTime())) return dtStr;
        return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    } catch (e) {
        return dtStr;
    }
}

function getStatusBadgeClass(status, canRemove = false) {
    if (status === 'Available') return 'status-available';
    if (status === 'SOLD_OUT') return canRemove ? 'status-eligible' : 'status-sold';
    if (status === 'REMOVED') return 'status-removed';
    return 'status-available';
}

function getOrderStatusBadgeClass(status) {
    if (!status) return 'status-available';
    const s = status.toUpperCase();
    if (s === 'AVAILABLE') return 'status-available';
    if (s === 'CONFIRMED') return 'status-confirmed';
    if (s === 'PURCHASED') return 'status-purchased';
    if (s === 'CANCELLED') return 'status-cancelled';
    if (s === 'SOLD_OUT') return 'status-sold';
    if (s === 'REMOVED') return 'status-removed';
    return 'status-available';
}

function getDeliveryStatusBadgeClass(status) {
    if (!status) return 'status-available';
    const s = status.toUpperCase();
    if (s === 'DELIVERED') return 'status-delivered';
    if (s === 'REJECTED') return 'status-rejected';
    return 'status-available';
}

function escapeQuotes(str) {
    if (!str) return '';
    return String(str).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function escapeHTML(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// =========================================================
// 9. ORDER ACTION MODALS (CANCEL, MARK PURCHASED, REJECT, REVIEW)
// =========================================================

// --- Cancel Order Modal ---
function openCancelOrderModal(orderId) {
    document.getElementById('cancel-order-id').value = orderId;
    document.getElementById('cancel-reason-select').value = '';
    document.getElementById('cancel-explanation-group').style.display = 'none';
    document.getElementById('cancel-explanation-text').value = '';
    document.getElementById('modal-cancel-order').style.display = 'flex';
}

function closeCancelOrderModal() {
    document.getElementById('modal-cancel-order').style.display = 'none';
    document.getElementById('form-cancel-order').reset();
}

function onCancelReasonChanged() {
    const sel = document.getElementById('cancel-reason-select').value;
    const expGroup = document.getElementById('cancel-explanation-group');
    const expText = document.getElementById('cancel-explanation-text');
    if (sel === 'Other') {
        expGroup.style.display = 'block';
        expText.required = true;
    } else {
        expGroup.style.display = 'none';
        expText.required = false;
    }
}

async function submitCancelOrder(e) {
    e.preventDefault();
    if (!currentUser) return;
    const orderId = document.getElementById('cancel-order-id').value;
    const reason = document.getElementById('cancel-reason-select').value;
    const details = document.getElementById('cancel-explanation-text').value.trim();

    if (!reason) {
        showToast('Please select a cancellation reason.', 'warning');
        return;
    }

    if (reason === 'Other' && !details) {
        showToast('Please provide a written explanation for cancellation.', 'warning');
        return;
    }

    const btn = document.getElementById('btn-submit-cancel-order');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Cancelling...';

    try {
        const res = await fetch(`/api/orders/${orderId}/cancel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: currentUser.id,
                cancellation_reason: reason,
                cancellation_details: details
            })
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-ban"></i> Confirm Cancellation';

        if (!res.ok) {
            showToast(data.error || 'Failed to cancel order.', 'error');
            return;
        }

        showToast(data.message || 'Order successfully cancelled. Livestock returned to marketplace.', 'success');
        closeCancelOrderModal();
        if (currentUser.role === 'buyer') {
            loadBuyerPurchasedLivestock();
            loadBuyerPurchaseHistory();
            fetchAvailableListings();
        } else if (currentUser.role === 'farmer') {
            loadFarmerHistory();
            loadFarmerListings();
        }
    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-ban"></i> Confirm Cancellation';
        showToast('Server error during order cancellation.', 'error');
    }
}

// --- Mark as Purchased Modal (Proof Photo Required) ---
let proofPhotoBlob = null;

function openMarkPurchasedModal(orderId) {
    document.getElementById('purchased-order-id').value = orderId;
    clearProofImage();
    document.getElementById('modal-mark-purchased').style.display = 'flex';
}

function closeMarkPurchasedModal() {
    document.getElementById('modal-mark-purchased').style.display = 'none';
    clearProofImage();
}

function triggerProofFileInput() {
    document.getElementById('proof-photo-file').click();
}

function handleProofPhotoChosen(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
        showToast('Please select a valid image (JPG, PNG, or WEBP).', 'warning');
        return;
    }

    proofPhotoBlob = file;
    const reader = new FileReader();
    reader.onload = (evt) => {
        document.getElementById('proof-img-preview').src = evt.target.result;
        document.getElementById('proof-dropzone-placeholder').style.display = 'none';
        document.getElementById('proof-img-preview-box').style.display = 'flex';
    };
    reader.readAsDataURL(file);
}

function clearProofImage(e) {
    if (e) e.stopPropagation();
    proofPhotoBlob = null;
    const fileInput = document.getElementById('proof-photo-file');
    if (fileInput) fileInput.value = '';
    const previewImg = document.getElementById('proof-img-preview');
    if (previewImg) previewImg.src = '';
    const previewBox = document.getElementById('proof-img-preview-box');
    if (previewBox) previewBox.style.display = 'none';
    const placeholder = document.getElementById('proof-dropzone-placeholder');
    if (placeholder) placeholder.style.display = 'block';
}

async function submitMarkPurchased(e) {
    e.preventDefault();
    if (!currentUser) return;
    const orderId = document.getElementById('purchased-order-id').value;

    if (!proofPhotoBlob) {
        showToast('Please upload a delivery proof photo to mark as purchased.', 'warning');
        return;
    }

    const btn = document.getElementById('btn-submit-mark-purchased');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Uploading proof...';

    const formData = new FormData();
    formData.append('user_id', currentUser.id);
    formData.append('proof_photo', proofPhotoBlob);

    try {
        const res = await fetch(`/api/orders/${orderId}/mark-purchased`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-check-double"></i> Upload Proof & Complete Purchase';

        if (!res.ok) {
            showToast(data.error || 'Failed to complete purchase.', 'error');
            return;
        }

        showToast('Livestock marked as PURCHASED and DELIVERED successfully!', 'success');
        closeMarkPurchasedModal();
        if (currentUser.role === 'buyer') {
            loadBuyerPurchasedLivestock();
            loadBuyerPurchaseHistory();
        } else if (currentUser.role === 'farmer') {
            loadFarmerSold();
            loadFarmerHistory();
        }
    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-check-double"></i> Upload Proof & Complete Purchase';
        showToast('Server error while submitting proof photo.', 'error');
    }
}

// --- Reject Delivery Modal ---
function openRejectDeliveryModal(orderId) {
    document.getElementById('reject-order-id').value = orderId;
    document.getElementById('form-reject-delivery').reset();
    document.getElementById('modal-reject-delivery').style.display = 'flex';
}

function closeRejectDeliveryModal() {
    document.getElementById('modal-reject-delivery').style.display = 'none';
    document.getElementById('form-reject-delivery').reset();
}

async function submitRejectDelivery(e) {
    e.preventDefault();
    if (!currentUser) return;
    const orderId = document.getElementById('reject-order-id').value;
    const reasonSel = document.getElementById('reject-reason-select').value;
    const reasonText = document.getElementById('reject-details-text').value.trim();

    if (!reasonSel) {
        showToast('Please select a rejection reason.', 'warning');
        return;
    }

    const fullReason = reasonText ? `${reasonSel} - ${reasonText}` : reasonSel;

    const btn = document.getElementById('btn-submit-reject-delivery');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Rejecting...';

    try {
        const res = await fetch(`/api/orders/${orderId}/reject-delivery`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: currentUser.id,
                rejection_reason: fullReason
            })
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Confirm Delivery Rejection';

        if (!res.ok) {
            showToast(data.error || 'Failed to reject delivery.', 'error');
            return;
        }

        showToast(data.message || 'Delivery marked as REJECTED. Listing returned to marketplace.', 'success');
        closeRejectDeliveryModal();
        if (currentUser.role === 'buyer') {
            loadBuyerPurchasedLivestock();
            loadBuyerPurchaseHistory();
            fetchAvailableListings();
        } else if (currentUser.role === 'farmer') {
            loadFarmerSold();
            loadFarmerHistory();
            loadFarmerListings();
        }
    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Confirm Delivery Rejection';
        showToast('Server error while rejecting delivery.', 'error');
    }
}

// --- Ratings & Reviews Modal ---
const RATING_LABELS = {
    1: '1 Star - Poor',
    2: '2 Stars - Fair',
    3: '3 Stars - Good',
    4: '4 Stars - Very Good',
    5: '5 Stars - Excellent'
};

function openReviewModal(orderId) {
    document.getElementById('review-order-id').value = orderId;
    document.getElementById('review-text-input').value = '';
    setReviewRating(5);
    document.getElementById('modal-order-review').style.display = 'flex';
}

function closeReviewModal() {
    document.getElementById('modal-order-review').style.display = 'none';
    document.getElementById('form-order-review').reset();
}

function setReviewRating(rating) {
    document.getElementById('review-rating-value').value = rating;
    const labelEl = document.getElementById('review-rating-label');
    if (labelEl) labelEl.textContent = RATING_LABELS[rating] || `${rating} Stars`;

    const stars = document.querySelectorAll('#star-rating-widget .star-btn');
    stars.forEach(star => {
        const r = parseInt(star.getAttribute('data-rating'));
        if (r <= rating) {
            star.classList.add('active');
        } else {
            star.classList.remove('active');
        }
    });
}

async function submitOrderReview(e) {
    e.preventDefault();
    if (!currentUser) return;
    const orderId = document.getElementById('review-order-id').value;
    const rating = parseInt(document.getElementById('review-rating-value').value) || 5;
    const review = document.getElementById('review-text-input').value.trim();

    const btn = document.getElementById('btn-submit-review');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Submitting...';

    try {
        const res = await fetch(`/api/orders/${orderId}/review`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                buyer_id: currentUser.id,
                rating,
                review
            })
        });
        const data = await res.json();
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Submit Feedback';

        if (!res.ok) {
            showToast(data.error || 'Failed to submit review.', 'error');
            return;
        }

        showToast('Thank you! Your rating and review have been submitted.', 'success');
        closeReviewModal();
        if (currentUser.role === 'buyer') {
            loadBuyerPurchasedLivestock();
            loadBuyerPurchaseHistory();
        }
    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Submit Feedback';
        showToast('Server error while submitting review.', 'error');
    }
}

// --- View Delivery Proof Modal ---
function openViewProofModal(photoUrl, caption) {
    const modal = document.getElementById('modal-view-proof');
    const img = document.getElementById('view-proof-img');
    const cap = document.getElementById('view-proof-caption');
    if (!modal || !img) return;

    img.src = photoUrl;
    if (cap) cap.textContent = caption || 'Delivery Proof Photo';
    modal.style.display = 'flex';
}

function closeViewProofModal() {
    const modal = document.getElementById('modal-view-proof');
    if (modal) modal.style.display = 'none';
}
