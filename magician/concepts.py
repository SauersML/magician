"""Concepts and the sentences their directions come from.

Each concept is (name, phrase, category): the name is how an instruction refers to it, the phrase fills the slot {X}
of every sentence in SENTENCES. A concept's direction at layer l is the mean residual over the slot span (from the
phrase to the end of the sentence) across all sentences, minus the same mean averaged over the training concepts.
Every fourth concept of each category (by list position, starting with the second) is held out of training.
"""

CONCEPTS = [
    # animals
    ('cat', 'cats', 'animal'), ('dog', 'dogs', 'animal'), ('horse', 'horses', 'animal'), ('rabbit', 'rabbits', 'animal'),
    ('elephant', 'elephants', 'animal'), ('lion', 'lions', 'animal'), ('tiger', 'tigers', 'animal'), ('bear', 'bears', 'animal'),
    ('wolf', 'wolves', 'animal'), ('fox', 'foxes', 'animal'), ('monkey', 'monkeys', 'animal'), ('snake', 'snakes', 'animal'),
    ('shark', 'sharks', 'animal'), ('whale', 'whales', 'animal'), ('dolphin', 'dolphins', 'animal'), ('eagle', 'eagles', 'animal'),
    ('owl', 'owls', 'animal'), ('penguin', 'penguins', 'animal'), ('frog', 'frogs', 'animal'), ('spider', 'spiders', 'animal'),
    ('bee', 'bees', 'animal'), ('butterfly', 'butterflies', 'animal'), ('cow', 'cows', 'animal'), ('pig', 'pigs', 'animal'),
    ('sheep', 'sheep', 'animal'), ('chicken', 'chickens', 'animal'), ('mouse', 'mice', 'animal'), ('deer', 'deer', 'animal'),
    # food and drink
    ('pizza', 'pizza', 'food'), ('bread', 'bread', 'food'), ('cheese', 'cheese', 'food'), ('chocolate', 'chocolate', 'food'),
    ('coffee', 'coffee', 'food'), ('tea', 'tea', 'food'), ('apple', 'apples', 'food'), ('banana', 'bananas', 'food'),
    ('strawberry', 'strawberries', 'food'), ('rice', 'rice', 'food'), ('pasta', 'pasta', 'food'), ('soup', 'soup', 'food'),
    ('cake', 'cake', 'food'), ('cookie', 'cookies', 'food'), ('sushi', 'sushi', 'food'), ('burger', 'burgers', 'food'),
    ('salad', 'salad', 'food'), ('honey', 'honey', 'food'), ('egg', 'eggs', 'food'), ('milk', 'milk', 'food'),
    ('wine', 'wine', 'food'), ('beer', 'beer', 'food'), ('tomato', 'tomatoes', 'food'), ('potato', 'potatoes', 'food'),
    # objects and vehicles
    ('car', 'cars', 'object'), ('bicycle', 'bicycles', 'object'), ('train', 'trains', 'object'), ('airplane', 'airplanes', 'object'),
    ('boat', 'boats', 'object'), ('chair', 'chairs', 'object'), ('lamp', 'lamps', 'object'), ('phone', 'phones', 'object'),
    ('computer', 'computers', 'object'), ('book', 'books', 'object'), ('pencil', 'pencils', 'object'), ('clock', 'clocks', 'object'),
    ('key', 'keys', 'object'), ('guitar', 'guitars', 'object'), ('piano', 'pianos', 'object'), ('camera', 'cameras', 'object'),
    ('umbrella', 'umbrellas', 'object'), ('hat', 'hats', 'object'), ('shoe', 'shoes', 'object'), ('bottle', 'bottles', 'object'),
    ('knife', 'knives', 'object'), ('mirror', 'mirrors', 'object'), ('candle', 'candles', 'object'), ('sword', 'swords', 'object'),
    # places
    ('France', 'France', 'place'), ('Japan', 'Japan', 'place'), ('Brazil', 'Brazil', 'place'), ('Egypt', 'Egypt', 'place'),
    ('India', 'India', 'place'), ('Canada', 'Canada', 'place'), ('Italy', 'Italy', 'place'), ('Russia', 'Russia', 'place'),
    ('China', 'China', 'place'), ('Mexico', 'Mexico', 'place'), ('Paris', 'Paris', 'place'), ('London', 'London', 'place'),
    ('Tokyo', 'Tokyo', 'place'), ('New York', 'New York', 'place'), ('Rome', 'Rome', 'place'), ('Africa', 'Africa', 'place'),
    ('the desert', 'the desert', 'place'), ('the forest', 'the forest', 'place'), ('mountains', 'mountains', 'place'),
    ('the ocean', 'the ocean', 'place'), ('rivers', 'rivers', 'place'), ('the beach', 'the beach', 'place'),
    ('islands', 'islands', 'place'), ('the moon', 'the moon', 'place'),
    # nature and weather
    ('rain', 'rain', 'nature'), ('snow', 'snow', 'nature'), ('storms', 'storms', 'nature'), ('thunder', 'thunder', 'nature'),
    ('the sun', 'the sun', 'nature'), ('stars', 'stars', 'nature'), ('fire', 'fire', 'nature'), ('ice', 'ice', 'nature'),
    ('wind', 'wind', 'nature'), ('clouds', 'clouds', 'nature'), ('rainbows', 'rainbows', 'nature'), ('volcanoes', 'volcanoes', 'nature'),
    ('earthquakes', 'earthquakes', 'nature'), ('flowers', 'flowers', 'nature'), ('trees', 'trees', 'nature'), ('gold', 'gold', 'nature'),
    # feelings
    ('love', 'love', 'feeling'), ('anger', 'anger', 'feeling'), ('fear', 'fear', 'feeling'), ('joy', 'joy', 'feeling'),
    ('sadness', 'sadness', 'feeling'), ('jealousy', 'jealousy', 'feeling'), ('pride', 'pride', 'feeling'), ('guilt', 'guilt', 'feeling'),
    ('hope', 'hope', 'feeling'), ('boredom', 'boredom', 'feeling'), ('loneliness', 'loneliness', 'feeling'), ('surprise', 'surprise', 'feeling'),
    ('calm', 'calm', 'feeling'), ('excitement', 'excitement', 'feeling'), ('shame', 'shame', 'feeling'), ('curiosity', 'curiosity', 'feeling'),
    # people
    ('doctors', 'doctors', 'person'), ('teachers', 'teachers', 'person'), ('soldiers', 'soldiers', 'person'), ('farmers', 'farmers', 'person'),
    ('pirates', 'pirates', 'person'), ('kings', 'kings', 'person'), ('queens', 'queens', 'person'), ('babies', 'babies', 'person'),
    ('wizards', 'wizards', 'person'), ('robots', 'robots', 'person'), ('astronauts', 'astronauts', 'person'), ('chefs', 'chefs', 'person'),
    ('police officers', 'police officers', 'person'), ('nurses', 'nurses', 'person'), ('artists', 'artists', 'person'),
    ('scientists', 'scientists', 'person'),
    # topics
    ('music', 'music', 'topic'), ('mathematics', 'mathematics', 'topic'), ('money', 'money', 'topic'), ('war', 'war', 'topic'),
    ('death', 'death', 'topic'), ('time', 'time', 'topic'), ('religion', 'religion', 'topic'), ('politics', 'politics', 'topic'),
    ('science', 'science', 'topic'), ('art', 'art', 'topic'), ('poetry', 'poetry', 'topic'), ('sports', 'sports', 'topic'),
    ('football', 'football', 'topic'), ('chess', 'chess', 'topic'), ('history', 'history', 'topic'), ('medicine', 'medicine', 'topic'),
    ('law', 'law', 'topic'), ('technology', 'technology', 'topic'), ('language', 'language', 'topic'), ('philosophy', 'philosophy', 'topic'),
    ('dance', 'dance', 'topic'), ('crime', 'crime', 'topic'), ('school', 'school', 'topic'), ('family', 'family', 'topic'),
    # colours
    ('red', 'the color red', 'color'), ('blue', 'the color blue', 'color'), ('green', 'the color green', 'color'),
    ('yellow', 'the color yellow', 'color'), ('purple', 'the color purple', 'color'), ('black', 'the color black', 'color'),
    ('white', 'the color white', 'color'), ('pink', 'the color pink', 'color'),
]

NAMES = [c[0] for c in CONCEPTS]
PHRASES = [c[1] for c in CONCEPTS]
CATEGORY = [c[2] for c in CONCEPTS]


def _held_out():
    seen = {}
    out = []
    for i, cat in enumerate(CATEGORY):
        j = seen.get(cat, 0)
        seen[cat] = j + 1
        if j % 4 == 1:
            out.append(i)
    return out


TEST_CONCEPTS = _held_out()
TRAIN_CONCEPTS = [i for i in range(len(CONCEPTS)) if i not in set(TEST_CONCEPTS)]

# Sentences with one slot {X} that read naturally for any concept phrase.
SENTENCES = [
    "I can't stop thinking about {X}.",
    "The whole lecture was about {X}.",
    "My favorite thing to talk about is {X}.",
    "She wrote a long poem about {X}.",
    "This picture reminds me of {X}.",
    "He has always been fascinated by {X}.",
    "Today's topic is {X}.",
    "The documentary focused on {X}.",
    "We spent the whole evening discussing {X}.",
    "There is a new book about {X}.",
    "Everyone in the room was talking about {X}.",
    "My dream last night was full of {X}.",
    "The painting on the wall showed {X}.",
    "Her notebook was full of drawings of {X}.",
    "The museum opened an exhibit on {X}.",
    "Nothing interests me more than {X}.",
    "The children asked a hundred questions about {X}.",
    "His essay on {X} won first prize.",
    "I searched the internet for {X}.",
    "The song is about {X}.",
    "Grandpa told us old stories about {X}.",
    "The podcast episode was all about {X}.",
    "They made a movie about {X}.",
    "My teacher knows everything about {X}.",
    "The magazine cover featured {X}.",
    "I read an article about {X} this morning.",
    "Our class project is on {X}.",
    "Somebody left a note about {X} on my desk.",
    "The comedian made jokes about {X}.",
    "Let me tell you about {X}.",
    "The professor gave a talk on {X}.",
    "Last summer I learned a lot about {X}.",
    "The poster in the hallway was about {X}.",
    "She can talk about {X} for hours.",
    "The quiz had ten questions about {X}.",
    "My little brother is obsessed with {X}.",
    "The library has a whole shelf on {X}.",
    "The tour guide kept mentioning {X}.",
    "In my diary I wrote about {X}.",
    "The game show category was {X}.",
    "We watched a video about {X}.",
    "Our neighbor collects pictures of {X}.",
    "The meeting turned into a debate about {X}.",
    "Mom bought me a puzzle with {X} on it.",
    "The headline was about {X}.",
    "A stranger on the bus talked to me about {X}.",
    "The final exam covered {X}.",
    "I drew a sketch of {X} in class.",
]
