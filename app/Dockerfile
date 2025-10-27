FROM nginx:alpine

# Copy static site
COPY ./static /usr/share/nginx/html

EXPOSE 3000

# Override default nginx config to listen on 3000
RUN sed -i 's/listen       80;/listen       3000;/g' /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]
