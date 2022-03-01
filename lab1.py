import re

# zad 1
text1 = "Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku"

removeNumbers = re.sub("[0-9]+", "", text1)
print(removeNumbers)

text2 = "<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a> </p></div>"
removeHTML = re.sub("[^>]", "", text2)
print(removeHTML)

text3 = """Lorem ipsum dolor sit amet, consectetur; adipiscing elit.
Sed eget mattis sem. Mauris egestas erat quam, ut faucibus eros congue et. In
blandit, mi eu porta; lobortis, tortor nisl facilisis leo, at tristique augue risus
eu risus."""
removePunctuaction = re.sub(r"[^\w\s]", '', text3)
print(removePunctuaction)

# zad 2
text4 = """Lorem ipsum dolor
sit amet, consectetur adipiscing elit. Sed #texting eget mattis sem. Mauris #frasista
egestas erat #tweetext quam, ut faucibus eros #frasier congue et. In blandit, mi eu porta
lobortis, tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus."""
findHashtags = re.findall(r"#[a-z0-9_]+", text4)
print(findHashtags)

# zad 3

text5 = """:) ;) ;( :> :< ;< :-) ;-)"""
findEmotes = re.findall(r"[:|;][-]?[\)|\(|<|>]", text5)
print(findEmotes)
