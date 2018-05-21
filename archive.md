---
layout: page
title: Archive
---

Chronological list of articles:

<ol>
  {% for archives in site.posts %}
    <li>
      <a href="{{site.url}}{{archives.url}}">{{archives.title}} : {{archives.date | date: "%-d %B %Y"}}</a>      
    </li>
  {% endfor %}
</ol>
