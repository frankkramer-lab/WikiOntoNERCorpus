FROM node:20

# Setup node app
COPY . /wtf
WORKDIR /wtf
RUN npm install && npm run build

# Increase V8 heap size
ENV NODE_OPTIONS=--max-old-space-size=16384
#CMD ["node", "./dist/app.js"]
ENTRYPOINT ["bash"]
CMD ["entrypoint.sh"]
