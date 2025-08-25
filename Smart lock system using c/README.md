# 🔐 Smart Lock System (C)

## 📌 Description
A Windows console-based **Smart Lock** written in C. It supports multiple doors, each with its own password and optional hint. The program enforces a basic password policy (first letter capital, length ≥ 8). On wrong attempts it triggers an audible alarm and flashes the console. After **5 failed attempts**, it shows a warning **popup** and runs a **10-second countdown**. 

---

## 🚀 Features
- 🚪 Multiple doors with individual passwords.
- 🧩 Optional hint per door.
- ✅ Password policy: first letter **capital** and length **≥ 8**.
- 🔔 Beep alarm on wrong attempts.
- 🚨 Console flash warning on wrong attempts.
- ⚠️ Popup alert + 10-second countdown after 5 failed tries.
- 🖥️ Pure C on Windows using WinAPI (Beep, MessageBox, console colors).

---

## 📂 Project Structure
```
├── lock.c     # Smart Lock implementation (Windows C console app)
└── README.md  # Project documentation
```

---

## ⚙️ Compile & Run (Windows)

### Option 1: MinGW (GCC)
```bash
gcc lock.c -o lock.exe -luser32
.\lock.exe
```

### Option 2: Microsoft Visual C++ (Developer Command Prompt)
```bat
cl lock.c user32.lib
lock.exe
```

> Notes:
> - `user32` is needed for `MessageBox`.  
> - `Beep`, `Sleep`, and console APIs resolve via Windows system libs.

---

## ▶️ Usage
1. Run the executable.
2. Enter **number of doors**.
3. For each door, set **password** (policy enforced) and choose whether to add a **hint**.
4. Enter **door number** to unlock and type the **password**.
5. On wrong attempts:
   - Beep + flashing warning.
   - At 5th wrong attempt: popup + 10-second countdown.

---

## 🧪 Example Session
```text
Enter number of doors: 2
Enter details (password and hint)
Door 1
Password: SecurePass1
Do you want to set a hint? (y/n): y
Hint: favColor

Door 2
Password: StrongKey8
Do you want to set a hint? (y/n): n

Enter door number: 1
Enter your password: wrongpass
***** WARNING! INCORRECT PASSWORD *****
Incorrect password. Type 'try' to retry or 'hint' to get a hint: hint
Your hint is: favColor
Enter your password: SecurePass1
Door is opening...
```

---

## ✅ Future Enhancements
- Persist door data to a file (save/load).
- Mask password input (no echo).
- Add lockout timer per door.
- Cross-platform version (abstract WinAPI calls).

---

## 👨‍💻 Authors
- Chinmay Rajesh Khiste (Group Leader) 
- Sharwill Kiran Khisti 
- Shraddha Prakash Khetmalis 
- Sairaj Ramesh Khot  
- Krishna Dinesh Khiraiya 
- Ritesh Vijay Khotale 
