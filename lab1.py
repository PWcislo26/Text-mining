import re

# zad 1
text1 = "Dzisiaj mamy 4 stopnie na plusie,  marca 2022 roku"
remove_numbers = re.sub("\d+", '', text1)
print(remove_numbers)

text2 = "<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a> </p></div>"
remove_html = re.sub("[<(\\)?>]", "", text2)
print(remove_html)

text3 = """Lorem ipsum dolor sit amet, consectetur; adipiscing elit.
Sed eget mattis sem. Mauris egestas erat quam, ut faucibus eros congue et. In
blandit, mi eu porta; lobortis, tortor nisl facilisis leo, at tristique augue risus
eu risus."""
remove_punctuaction = re.sub(r"[^\w\s]", '', text3)
print(remove_punctuaction)

# zad 2
text4 = """Lorem ipsum dolor
sit amet, consectetur adipiscing elit. Sed #texting eget mattis sem. Mauris #frasista
egestas erat #tweetext quam, ut faucibus eros #frasier congue et. In blandit, mi eu porta
lobortis, tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus."""
find_hashtags = re.findall(r"#[a-z0-9_]+", text4)
print(find_hashtags)

# zad 3

text5 = """:) ;) ;( :> :< ;< :-) ;-)"""
find_emotes = re.findall(r"[:|;][-]?[\)|\(|<|>]", text5)
print(find_emotes)
