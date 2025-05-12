import pandas as pd
import random
import csv

def generate_sample_data(filename="accident_data.csv", num_records=1000):
    data = []

    for _ in range(num_records):
        age = random.randint(18, 70)
        experience = random.randint(0, age - 18)  # нельзя иметь больше опыта, чем возраст-18
        car_type = random.choice(["sedan", "suv", "truck", "sports", "minivan"])
        weather = random.choice(["sunny", "rainy", "foggy", "snowy"])
        time_of_day = random.choice(["morning", "afternoon", "evening", "night"])
        speeding = random.choice([0, 1])  # 1 = превышал скорость
        seatbelt = random.choice([0, 1])  # 1 = был пристёгнут
        alcohol = random.choice([0, 1])
        accident = 1 if (speeding or alcohol) and random.random() < 0.7 else 0

        data.append([
            age, experience, car_type, weather, time_of_day,
            speeding, seatbelt, alcohol, accident
        ])

    df = pd.DataFrame(data, columns=[
        "age", "experience", "car_type", "weather", "time_of_day",
        "speeding", "seatbelt", "alcohol", "accident"
    ])

    df.to_csv(filename, index=False)
    print(f"[✔] Данные сохранены в {filename}")


if __name__ == "__main__":
    generate_sample_data()
