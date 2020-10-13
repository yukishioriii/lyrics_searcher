import scrapy
import re


class AnimeSongLyricsSpider(scrapy.Spider):
    name = "animesonglyrics"
    base_url = "https://www.animesonglyrics.com/"

    def start_requests(self):
        for i in range(2017, 2015, -1):
            for j in ["Summer", "Winter", "Fall", "Spring"]:
                yield scrapy.Request(url=f"https://www.animesonglyrics.com/tags/{j} {i}", callback=self.parse_anime)

    def parse_anime(self, response):
        anime_titles = set(response.xpath(
            "//div[@id='titlelist']/span[@class='homesongs']/a/@href").getall())
        song_titles = set(response.xpath(
            "//div[@id='songlist']/span[@class='homesongs']/a/@href").getall())
        for i in anime_titles:
            yield scrapy.Request(url=i, callback=self.parse_song)
        for i in song_titles:
            yield scrapy.Request(url=i, callback=self.parse_lyrics)

    def parse_song(self, response):
        song_titles = set([i for i in response.xpath(
            "//div[@id='songlist']/a/@href").getall() if len(i) > 10])
        for i in song_titles:
            yield scrapy.Request(url=i, callback=self.parse_lyrics)

    def parse_lyrics(self, response):
        en_text = "\n".join([i.strip() for i in response.xpath(
            "//div[@id='tab1']/text()").getall()])
        ja_text = "\n".join([i.strip() for i in response.xpath(
            "//div[@id='tab3']/text()").getall()])
        filename = "||".join(response.url.split("/")[-2:])
        filename = re.sub("[^A-z0-9]", "_", filename)
        with open(f'animesonglyrics/{filename}.txt', 'w+') as f:
            f.write(en_text + "\n\n" + ja_text)
