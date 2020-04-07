Bootstrap: docker
From: dylanturpin/6-pack:CIRCLE_SHA1

%post
  mkdir /h
  chmod -R 777 /h
  chmod -R 777 /root
  chmod -R 777 /opt
  chmod -R 777 /etc
  chmod -R 777 /var
  chmod -R 777 /dev

%runscript
  echo helloworld
