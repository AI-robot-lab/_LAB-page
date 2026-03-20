# Bash i dostęp do robota (NVIDIA Jetson) po SSH

Ten artykuł wprowadza nowych członków zespołu do pracy terminalowej oraz bezpiecznego łączenia się z modułem NVIDIA Jetson zainstalowanym w robocie. W praktyce większość diagnostyki, uruchamiania skryptów i konfiguracji usług na robocie odbywa się właśnie przez Bash i SSH.

---

## 1. Po co nam Bash i SSH?

### Bash

Bash to powłoka tekstowa, która pozwala:

- uruchamiać programy i skrypty,
- zarządzać plikami,
- monitorować procesy,
- konfigurować środowisko ROS2,
- automatyzować powtarzalne zadania.

### SSH

SSH (*Secure Shell*) umożliwia bezpieczne połączenie zdalne z Jetsonem, dzięki czemu możesz:

- zalogować się do robota z własnego laptopa,
- kopiować pliki,
- uruchamiać węzły ROS2,
- przeglądać logi,
- diagnozować problemy bez podłączania monitora i klawiatury do robota.

---

## 2. Podstawy pracy w Bash

### 2.1. Nawigacja po katalogach

```bash
pwd
ls
ls -la
cd ~/projekty
cd ..
```

Najważniejsze skróty:

- `~` — katalog domowy użytkownika,
- `.` — bieżący katalog,
- `..` — katalog wyżej.

### 2.2. Operacje na plikach

```bash
mkdir -p ~/workspace/robot-tests
cp config.yaml backup/config.yaml
mv notes.txt notes-old.txt
rm file.tmp
rm -r old_logs
```

> `rm` usuwa pliki bez kosza. Używaj ostrożnie.

### 2.3. Podgląd plików i logów

```bash
cat README.md
less ~/.bashrc
head -n 20 log.txt
tail -n 50 log.txt
tail -f /var/log/syslog
```

### 2.4. Procesy i diagnostyka

```bash
ps aux | grep ros
htop
uname -a
whoami
hostname
```

---

## 3. Przygotowanie do połączenia SSH

Przed pierwszym logowaniem potrzebujesz:

- konta użytkownika na Jetsonie,
- adresu IP lub nazwy hosta robota,
- dostępu do tej samej sieci,
- najlepiej skonfigurowanego klucza SSH.

### 3.1. Sprawdzenie, czy masz klucz SSH

```bash
ls -la ~/.ssh
```

Jeśli nie masz jeszcze pary kluczy, utwórz ją:

```bash
ssh-keygen -t ed25519 -C "imie.nazwisko@prz.edu.pl"
```

Najczęściej możesz zaakceptować domyślną ścieżkę:

- klucz prywatny: `~/.ssh/id_ed25519`
- klucz publiczny: `~/.ssh/id_ed25519.pub`

### 3.2. Wgranie klucza publicznego na Jetsona

Jeżeli administrator podał tymczasowe hasło, dodaj klucz poleceniem:

```bash
ssh-copy-id user@192.168.1.50
```

Jeśli `ssh-copy-id` nie działa, można zrobić to ręcznie:

```bash
cat ~/.ssh/id_ed25519.pub
```

Skopiuj wynik do pliku `~/.ssh/authorized_keys` na Jetsonie.

---

## 4. Logowanie do Jetsona

### 4.1. Podstawowe połączenie

```bash
ssh user@192.168.1.50
```

Przy pierwszym połączeniu zobaczysz pytanie o fingerprint hosta. Jeśli adres został potwierdzony przez opiekuna sprzętu, wpisz:

```text
yes
```

### 4.2. Wygodny alias w `~/.ssh/config`

Aby nie wpisywać za każdym razem pełnego adresu, dodaj konfigurację:

```bash
cat >> ~/.ssh/config << 'EOF'
Host jetson-robot
    HostName 192.168.1.50
    User user
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 4
EOF
```

Wtedy łączysz się krócej:

```bash
ssh jetson-robot
```

---

## 5. Najczęstsze czynności po zalogowaniu

### 5.1. Sprawdzenie podstawowego stanu systemu

```bash
hostname
whoami
pwd
df -h
free -h
```

### 5.2. Monitorowanie zasobów Jetsona

```bash
tegrastats
```

To jedno z podstawowych narzędzi na platformie NVIDIA Jetson — pokazuje m.in. użycie CPU, GPU, pamięci i temperatury.

### 5.3. Uruchamianie środowiska ROS2

W wielu projektach trzeba załadować środowisko:

```bash
source /opt/ros/humble/setup.bash
source ~/workspace/robot_ws/install/setup.bash
```

Po tym możesz sprawdzać tematy i węzły:

```bash
ros2 topic list
ros2 node list
```

---

## 6. Kopiowanie plików między laptopem a Jetsonem

### 6.1. Wysyłanie pliku na robota

```bash
scp config.yaml jetson-robot:~/workspace/robot_ws/
```

### 6.2. Pobieranie logów z Jetsona

```bash
scp jetson-robot:~/logs/session-01.txt ./logs/
```

### 6.3. Kopiowanie całego katalogu

```bash
scp -r ./models jetson-robot:~/workspace/robot_ws/assets/
```

---

## 7. Przydatne polecenia zdalne

Możesz wykonać pojedynczą komendę bez pełnego logowania:

```bash
ssh jetson-robot "hostname && uptime"
```

Lub uruchomić restart usługi:

```bash
ssh jetson-robot "systemctl --user restart perception.service"
```

---

## 8. Typowe problemy i rozwiązania

### 8.1. `Permission denied (publickey)`

Sprawdź:

- czy używasz poprawnego użytkownika,
- czy klucz publiczny jest w `authorized_keys`,
- czy agent SSH ma załadowany klucz,
- czy wpis w `~/.ssh/config` wskazuje dobry `IdentityFile`.

Pomocne komendy:

```bash
ssh -v jetson-robot
ssh-add -l
```

### 8.2. `Connection timed out`

Możliwe przyczyny:

- Jetson jest wyłączony,
- robot nie jest w tej samej sieci,
- zmienił się adres IP,
- firewall blokuje ruch.

Warto sprawdzić:

```bash
ping 192.168.1.50
```

### 8.3. Zbyt wolna sesja

Przyczyną może być słabe Wi‑Fi, duże obciążenie Jetsona albo nadmiar logów. W takiej sytuacji:

- ogranicz równoległe procesy,
- sprawdź `tegrastats`,
- nie uruchamiaj ciężkich modeli bez potrzeby,
- loguj dane do plików zamiast zasypywać terminal.

---

## 9. Minimalny workflow dla nowej osoby

1. Połącz się z siecią robota.
2. Sprawdź, czy działa `ping` do Jetsona.
3. Zaloguj się przez `ssh jetson-robot`.
4. Załaduj środowisko ROS2.
5. Uruchom podstawową diagnostykę (`ros2 topic list`, `tegrastats`).
6. Skopiuj potrzebne pliki przez `scp`.
7. Zapisz w notatkach, na jakim hoście i branchu pracowałeś.

---

## 10. Dobre praktyki bezpieczeństwa

- Nigdy nie udostępniaj klucza prywatnego SSH.
- Nie zapisuj haseł w repozytorium.
- Nie uruchamiaj komend z `sudo`, jeśli nie rozumiesz ich skutków.
- Zanim zrestartujesz usługę lub komputer pokładowy, upewnij się, że nikt inny nie prowadzi testu.
- Dokumentuj zmiany konfiguracyjne wykonane bezpośrednio na Jetsonie.

---

## Powiązane artykuły

- [Praca w zespole inżynierskim](#wiki-praca-w-zespole)
- [Git i GitHub](#wiki-git-github)
- [ROS2](#wiki-ros2)
- [Docker dla robotyki](#wiki-docker)

## Zasoby

- NVIDIA Jetson Linux Developer Guide
- OpenSSH Manual Pages
- Dokumentacja projektu i instrukcje opiekuna robota

---
*Ostatnia aktualizacja: 2026-03-20*
*Autor: Codex / Zespół Laboratorium*
