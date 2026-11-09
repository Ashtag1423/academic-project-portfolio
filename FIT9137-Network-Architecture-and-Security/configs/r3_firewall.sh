#!/bin/bash
	# Place the firewal settings here using iptables commands
	# This is a shell script and hash character makes the rest of the lin 
	# as comment
	
	# Drop any packet that doesn't match an explicit ACCEPT rule
	iptables -P INPUT   DROP
	iptables -P FORWARD DROP
	iptables -P OUTPUT  DROP
	
	# These are the most important rules. They allow return traffic for any
	# connection that we have authorized, drastically simplifying the ruleset.
	iptables -A INPUT   -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
	iptables -A OUTPUT  -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
	iptables -A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
	
	
	## task 1. 
	# Allows HTTP traffic to the Web server from ANY source
	iptables -A FORWARD -o eth0 -d 135.232.159.11 -p tcp --dport 80 -j ACCEPT
	
	# Allows SMTP traffic to the Mail server from ANY source
	iptables -A FORWARD -o eth0 -d 135.232.159.12 -p tcp --dport 25 -j ACCEPT
	
	# Allow TCP traffic for DNS from any source
	iptables -A FORWARD -o eth0 -d 135.232.159.10 -p tcp --dport 53 -j ACCEPT
	
	# Allow UDP traffic for DNS from any source
	iptables -A FORWARD -o eth0 -d 135.232.159.10 -p udp --dport 53 -j ACCEPT
	
	
	## TASK C 2
	# Allow the DNS server to initiate DNS queries to the internet
	iptables -A FORWARD -i eth0 -o eth3 -s 135.232.159.10 -p udp --dport 53 -m conntrack --ctstate NEW -j ACCEPT
	iptables -A FORWARD -i eth0 -o eth3 -s 135.232.159.10 -p tcp --dport 53 -m conntrack --ctstate NEW -j ACCEPT
	
	# Allow the Mail server to initiate SMTP connections to other mail servers
	iptables -A FORWARD -i eth0 -o eth3 -s 135.232.159.12 -p tcp --dport 25 -m conntrack --ctstate NEW -j ACCEPT
	
	# Allow the Web server to initiate HTTP/HTTPS connections for system updates
	iptables -A FORWARD -i eth0 -o eth3 -s 135.232.159.11 -p tcp --dport 80 -m conntrack --ctstate NEW -j ACCEPT
	iptables -A FORWARD -i eth0 -o eth3 -s 135.232.159.11 -p tcp --dport 443 -m conntrack --ctstate NEW -j ACCEPT
	
	
	## TASK C 3, ALLOW ALL SERVICE FOR INTERNAL USERS
	# Allow internal network on eth1 to access anything in the DMZ
	iptables -A FORWARD -i eth1 -o eth0 -m conntrack --ctstate NEW -j ACCEPT
	
	# Allow internal network on eth2 to access anything in the DMZ
	iptables -A FORWARD -i eth2 -o eth0 -m conntrack --ctstate NEW -j ACCEPT
	
	## TASK C 4
	# Allow traffic from the first internal network to the second
	iptables -A FORWARD -i eth1 -o eth2 -j ACCEPT
	
	# Allow traffic from the second internal network to the first
	iptables -A FORWARD -i eth2 -o eth1 -j ACCEPT
	
	## TASK C 5, ALLOW NEW CONNECTION OUT, AND TRAFFIC RELATED IN
	# Allow new connections originating from the first internal network to the Internet
	iptables -A FORWARD -i eth1 -o eth3 -m conntrack --ctstate NEW -j ACCEPT
	
	# Allow new connections originating from the second internal network to the Internet
	iptables -A FORWARD -i eth2 -o eth3 -m conntrack --ctstate NEW -j ACCEPT
	
	
	## TASK C 6, 
	# Allow NEW SSH connections TO THE ROUTER from the admin subnet
	iptables -A INPUT -s 135.232.27.0/24 -p tcp --dport 22 -m conntrack --ctstate NEW -j ACCEPT
	
	## TASK C 7
	
	## INPUT rules
	# Allow ICMP echo-requests TO the router from the DMZ network
	iptables -A INPUT -i eth0 -p icmp --icmp-type echo-request -j ACCEPT
	
	# Allow ICMP echo-requests TO the router from the first internal network
	iptables -A INPUT -i eth1 -p icmp --icmp-type echo-request -j ACCEPT
	
	# Allow ICMP echo-requests TO the router from the second internal network
	iptables -A INPUT -i eth2 -p icmp --icmp-type echo-request -j ACCEPT
	
	
	# output rules
	# Allow ICMP echo-requests FROM the router to the DMZ network
	iptables -A OUTPUT -o eth0 -p icmp --icmp-type echo-request -j ACCEPT
	
	# Allow ICMP echo-requests FROM the router to the first internal network
	iptables -A OUTPUT -o eth1 -p icmp --icmp-type echo-request -j ACCEPT
	
	# Allow ICMP echo-requests FROM the router to the second internal network
	iptables -A OUTPUT -o eth2 -p icmp --icmp-type echo-request -j ACCEPT
