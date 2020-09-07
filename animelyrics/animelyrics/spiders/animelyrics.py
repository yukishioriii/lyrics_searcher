import scrapy


class QuotesSpider(scrapy.Spider):
    name = "animelyrics"
    base_url = "https://www.animelyrics.com/"

    def start_requests(self):
        yield scrapy.Request(url="https://www.animelyrics.com/anime/", callback=self.parse_title)

    def parse_title(self, response):
        for url in response.xpath('//table//td/a/@href').getall():
            yield scrapy.Request(url=self.base_url + url, callback=self.parse_song_title)

    def parse_song_title(self, response):
        for element in response.xpath("//table[@class='mytable']//a"):
            url = element.xpath("./@href").get()
            url = url[:-3] + "jis" if url.endswith("htm") else url
            yield scrapy.Request(url=self.base_url + url, callback=self.parse_song_lyrics, meta={"song_title": element.xpath("./text()").get()})

    def parse_song_lyrics(self, response):
        title = response.meta.get('song_title')
        lyrics = "".join(response.xpath("//div[@id='kanji']//text()").getall())
        with open(f"{title}.txt", 'w+') as f:
            f.write(lyrics)
