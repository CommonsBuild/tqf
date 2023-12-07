import panel as pn
from item_list.item import Item
from item_list.item_list import ItemList

pn.extension(nthreads=0)

item1 = Item(value=5)
item2 = Item(value=10)

item_list = ItemList(items=[item1, item2])

tabs = pn.Tabs(
    ('Item 1', item1.view()),
    ('Item 2', item2.view()),
    ('Item list', item_list.view()),
    active=0,
    dynamic=True,
)

tabs.servable()
