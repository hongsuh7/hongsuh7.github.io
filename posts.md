---
layout: default
title: Posts
---
# Blog posts
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}" style="font-size: 24px">{{ post.title }}</a>
      <br>
      <p style="font-size: 14px">{{ post.date | date: '%B %d, %Y' }} </p>
      {{ post.excerpt }}
      <hr>
    </li>
  {% endfor %}
</ul>