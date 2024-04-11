import ccxt
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.model_selection import train_test_split
import time

# Download all btc 1h data
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
since = exchange.parse8601('2024-01-01T00:00:00Z')
all_data = []
while since < exchange.milliseconds():
    ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since)
    if len(ohlcvs) > 0:
        since = ohlcvs[-1][0] + 1
        all_data += ohlcvs
    else:
        break
df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df_train, df_test = train_test_split(df, test_size=0.4, shuffle=False)

mapping = {
    'O': 'open',
    'H': 'high',
    'L': 'low',
    'C': 'close'
}


class Rule:
    def __init__(self, look_back):
        self.left_member = (None, None)
        self.right_member = (None, None)
        self.operator = None
        self.look_back = look_back

    def __str__(self):
        return f'{self.left_member[0]}[{self.left_member[1]}] {self.operator} {self.right_member[0]}[{self.right_member[1]}]'

    def init_random(self):
        self.left_member = (np.random.choice(['O', 'H', 'L', 'C']), np.random.randint(0, self.look_back))
        self.right_member = (np.random.choice(['O', 'H', 'L', 'C']), np.random.randint(0, self.look_back))
        self.operator = np.random.choice(['<', '>'])

        # check for invalid rules
        if self.left_member[1] == self.right_member[1]:
            self.init_random()

    def evaluate(self, df):
        left_data = df[mapping[self.left_member[0]]].shift(self.left_member[1])
        right_data = df[mapping[self.right_member[0]]].shift(self.right_member[1])
        if self.operator == '<':
            result = left_data < right_data
        elif self.operator == '>':
            result = left_data > right_data
        return result


class Pattern:
    def __init__(self, pattern_size, look_back):
        self.pattern_size = pattern_size
        self.look_back = look_back
        self.fitness = 0
        self.rules = []

    def __str__(self):
        return ' & '.join([str(rule) for rule in self.rules])

    def init_random_patterns(self):
        for i in range(self.pattern_size):
            rule = Rule(self.look_back)
            rule.init_random()
            self.rules.append(rule)

        # check for invalid patterns
        if len(set([str(rule) for rule in self.rules])) != len(self.rules):
            self.rules = []
            self.init_random_patterns()

        # check for invalid patterns (e.g. O[0] < C[0] & C[0] > O[0]) you can not compare the same candle
        for i in range(self.pattern_size):
            if self.rules[i].left_member[1] == self.rules[i].right_member[1]:
                self.rules = []
                self.init_random_patterns()
                break

    def evaluate_pattern(self, df):
        results = [rule.evaluate(df) for rule in self.rules]
        result = results[0]
        for i in range(1, self.pattern_size):
            result = result & results[i]
        return result


class Population:
    def __init__(self,
                 pattern_size,
                 look_back,
                 population_size,
                 top_k,
                 mutation_rate=0.1,
                 crossover_rate=0.1,
                 n_generations=1000):
        self.pattern_size = pattern_size
        self.look_back = look_back
        self.population_size = population_size
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_generations = n_generations
        self.best_pattern = None
        self.patterns = []
        self.init_random_population()

    def init_random_population(self):
        self.patterns = []
        for _ in range(self.population_size):
            pattern = Pattern(self.pattern_size, self.look_back)
            pattern.init_random_patterns()
            self.patterns.append(pattern)

    def evaluate_population(self, df):
        for pattern in self.patterns:
            fitness = np.sum(df['close'].pct_change().shift(-1) * pattern.evaluate_pattern(df))
            pattern.fitness = fitness if fitness > 0 else 0
        self.patterns = sorted(self.patterns, key=lambda x: x.fitness, reverse=True)

    def select_by_rolette_wheel(self):
        fitnesses = [pattern.fitness for pattern in self.patterns]
        total_fitness = np.sum(fitnesses)
        if total_fitness == 0:
            selected_indices = np.random.choice(np.arange(self.population_size), size=self.top_k, replace=False)
        else:
            fitnesses = [fitness + 1e-10 for fitness in fitnesses]
            total_fitness += 1e-10 * len(fitnesses)
            probabilities = [fitness / total_fitness for fitness in fitnesses]
            selected_indices = np.random.choice(np.arange(self.population_size), size=self.top_k, p=probabilities,
                                                replace=False)
        selected_patterns = [self.patterns[i] for i in selected_indices]
        return selected_patterns

    def crossover(self, best_patterns):
        parent1, parent2 = np.random.choice(best_patterns, size=2, replace=False)
        crossover_point = np.random.randint(0, self.pattern_size)
        child1 = Pattern(self.pattern_size, self.look_back)
        child2 = Pattern(self.pattern_size, self.look_back)
        child1.rules = parent1.rules[:crossover_point] + parent2.rules[crossover_point:]
        child2.rules = parent2.rules[:crossover_point] + parent1.rules[crossover_point:]
        # Validate children they should have valid rules
        if len(set([str(rule) for rule in child1.rules])) != len(child1.rules) or len(
                set([str(rule) for rule in child2.rules])) != len(child2.rules):
            return self.crossover(best_patterns)

        return child1, child2

    def mutate(self, pattern):
        for rule in pattern.rules:
            if np.random.rand() < self.mutation_rate:
                rule.init_random()

    def evolve(self, df):
        for i in range(self.n_generations):
            time.sleep(0.5)
            self.evaluate_population(df)
            best_patterns = self.select_by_rolette_wheel()
            new_patterns = [copy.deepcopy(pattern) for pattern in best_patterns]
            max_fitness_pattern = max(self.patterns, key=lambda x: x.fitness)

            if self.best_pattern is None or max_fitness_pattern.fitness > self.best_pattern.fitness:
                self.best_pattern = copy.deepcopy(max_fitness_pattern)

            if self.best_pattern is not None:
                new_patterns.append(copy.deepcopy(self.best_pattern))

            if np.random.rand() < self.crossover_rate:
                while len(new_patterns) < self.population_size:
                    child1, child2 = self.crossover(new_patterns)
                    new_patterns += [child1, child2]
                new_patterns = new_patterns[:self.population_size]

            for pattern in new_patterns:
                self.mutate(pattern)

            if len(new_patterns) < self.population_size:
                for _ in range(self.population_size - len(new_patterns)):
                    new_pattern = Pattern(self.pattern_size, self.look_back)
                    new_pattern.init_random_patterns()
                    new_patterns.append(new_pattern)
            elif len(new_patterns) > self.population_size:
                new_patterns = new_patterns[:self.population_size]

            self.patterns = new_patterns
            print(f'Generation {i + 1}, best fitness: {self.best_pattern.fitness}')

        return self.best_pattern

population = Population(pattern_size=12, look_back=3, population_size=50, top_k=10, mutation_rate=0.25, crossover_rate=0.25, n_generations=150)
population.init_random_population()
best_pattern = population.evolve(df_train)
print(best_pattern)