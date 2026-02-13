import bcrypt

password = input("Enter password to hash: ").encode("utf-8")
hashed = bcrypt.hashpw(password, bcrypt.gensalt())

print("\nYour hashed password:\n")
print(hashed.decode("utf-8"))
