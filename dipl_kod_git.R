# Machine learning base classification of the spine
# Based on spine parameters, Decision tree, SVM, random forest and Naive Bayes models were formed
# 

getwd()
setwd("C:/Users/tesic/jelena/fakultet/BMI/STATISTIKA/DOMACI/domaci10/")
library(readxl)
library(tidyr)
## Importing packages
library(tidyverse)
library(MASS)
library(class)

C1 = read.csv("C:/Users/tesic/JELENA/fakultet/BMI/STATISTIKA/DOMACI/domaci1?/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv", stringsAsFactors = F)


colnames(C1)[colnames(C1) == "pelvic_tilt.numeric"] = "nagib_karlice"
colnames(C1)[colnames(C1) == "pelvic_incidence"] = "incidenca_karlice"
colnames(C1)[colnames(C1) == "lumbar_lordosis_angle"] = "lumbalna_lordoza"
colnames(C1)[colnames(C1) == "sacral_slope"] = "krsni_ugib"
colnames(C1)[colnames(C1) == "pelvic_radius"] = "radijus_karlice"
colnames(C1)[colnames(C1) == "degree_spondylolisthesis"] = "stepen_spondilolist?ze"
colnames(C1)[colnames(C1) == "class"] = "status"
View(C1)
summary(C1)


#sredjujemo set, za svaki slucaj stavljamo da su numeric
#C1$incidenca_karlice <- as.numeric(C1$incidenca_karlice)
#C1$nagib_karlice <- as.numeric(C1$nagib_karlice)
#C1$lumbalna_l?rdoza <- as.numeric(C1$lumbalna_lordoza)
#C1$krsni_ugib <- as.numeric(C1$krsni_ugib)
#C1$radijus_karlice <- as.numeric(C1$radijus_karlice)
#C1$stepen_spondilolisteze <- as.numeric(C1$stepen_spondilolisteze)

#bri�emo sve pripadnike koji u nekoj koloni imaj? NA vrednosti za kolone u setu koje nisu sting
C1 = C1 %>%
  na.omit() %>% 
  as.data.frame(stringsAsFactors = FALSE)


#vizuelizacija uzorka
plot_C1 = C1
C1$status = as.factor(C1$status) #status je kategoricka promenljiva
head(C1)
C1 %>%
  ggplot(aes(stat?s)) + geom_bar(fill = c("orange" , "darkgreen")) +
  ggtitle("Broj uzoraka u svako grupi")

#delimo podatke, u dve grupe na osnovu statusa, 0 i 1
plot_C1$status[plot_C1$status == "Abnormal"] = 1
plot_C1$status[plot_C1$status == "Normal"] = 0
#status ponovo?prevodimo sada u numericku, kontinualnu promenljivu
plot_C1$status = as.numeric(plot_C1$status)

#pravimo grupni grafik, u kom mo�emo videti kako nam se pona�aju svi podaci, jedni 
#u odnosu na druge
pairs(plot_C1, col = (plot_C1$status + 1))
#po�to pravim? kategoricki model, odnosno pravimo model koji ce moci da svrstava na�e podatke
#u kategorije na osnovu drugih podataka, bitno nam je da znamo kako se na�i podaci pona�aju 



# Distribucija stepen_spondilolisteze
library(ggplot2)
ggplot(C1, 
       aes(x ? stepen_spondilolisteze)) + 
  geom_histogram(fill = "orange2", 
                 col = "white") +
  geom_vline(aes(xintercept = median(C1$stepen_spondilolisteze)), 
             col = "red",
             size = 1) +
  labs(title = "Distribucija stepena sp?ndilolisteze - right skewed",
       x = "vrednosti uglova",
       y = "Frekvencija") +
  coord_cartesian(xlim = c(-13, 500))
#proveravamo autlejer koji vidimo da se izdvojio
which.max(C1$stepen_spondilolisteze)
#po�to je jedan, obradicemo ga, i dati mu s?ednju vrednost ove vrijable
ind <- which.max(C1$stepen_spondilolisteze)
C1[ind, ]
C1$stepen_spondilolisteze[ind] <- mean(C1$stepen_spondilolisteze)
summary(C1$stepen_spondilolisteze)
# distribution 
ggplot(C1, 
       aes(x = stepen_spondilolisteze)) + 
  ?eom_histogram(fill = "orange2", 
                 col = "white") +
  geom_vline(aes(xintercept = median(C1$stepen_spondilolisteze)), 
             col = "red",
             size = 1) +
  labs(title = "Distribution of stepen_spondilolisteze - right skewed",?       x = "Ocitane vrednosti",
       y = "Vrekvencija") +
  coord_cartesian(xlim = c(-13, 150))
#dobili smo malo bolju distribuciju od prethodne, tako da cemo je ostaviti takvu

#korelacija
suppressMessages(library(corrplot))
corr_mat <- cor(C1[,1:6])
co?rplot(corr_mat, method = "number", col =c ("goldenrod1", "darkorange1", "darkred") )


# MODELS

# NORMALIZAION
library(caret)
#C1_1 = as.data.frame(scale(C1[,1:6]))
#S = C1$status
#C1 = cbind(C1_1, S)
#colnames(C1)[colnames(C1) == "S"] =?"status"
#View(C1)
#summary(C1)

library(randomForest)
library(caTools)

# SPLITING DATA SET ON TRAIN AND TEST
set.seed(1234)
split = sample.split(C1$status, SplitRatio = 0.75)
train_data = subset(C1, split == TRUE)
test_data= subset(C1, split == FALSE)

?rain_data[-7] <- scale(train_data[-7])
test_data[-7] <- scale(test_data[-7])
str(test_data)


##DECISION TREE
library(rpart)
C1_tree <- rpart(status~., data=train_data)
#�tampamo da bismo videli optimalan broj podeonih granica
print(C1_tree$cptable)
librar?(rpart.plot)
rpart.plot(C1_tree, type = 2, fallen.leaves = F, extra = 2)

#otkrivamo:
#1.Nijedan od podeoka 5 (5. red) nema najni�i �xerror-0.5428571�.Tako da biramo tu 
#graicu da presecemo na�e drvo. �to mi nije najsjajnije re�enje
#Na�e drvo nam govori ?a je najznacajniji koren varijable stepen_spondilolisteze, koje ima
#presek na >=20, pri cemu je onda kicma normalna. U drugom slucaju, <20 je abnormalna.
#Onda se spu�tamo na sledeci cvor.

#Now we cut the tree
#CV error min at 4th split - row 4
cp <- min(C1_tree$cptable[4, ])
prune.tree.C1 <- prune(C1_tree, cp = cp)
rpart.plot(prune.tree.C1, type = 2, fallen.leaves = F, extra = 2)

rparty.test = predict(prune.tree.C1, newdata=test_data, type="class")
df1<-data.frame(Orig=test_data$status, Pred= rparty.test?
library(caret)
confusionMatrix(table(df1$Orig,df1$Pred))
#specificnost je za nijansu veca od senzitivnosti. iako nije mnogo veca, bilo bi nam korisnije
#da je senzitivnost veca

############RANDOM FOREST
rf.C1 <-randomForest(as.factor(status) ~., data=tra?n_data)
print(rf.C1)
# the error is higher when predicting the normal ones 

#vizuelizacija
plot(rf.C1)
t<- tuneRF(train_data[,-7], as.factor(train_data[,7]),
           stepFactor = 0.5, plot = TRUE, ntreeTry = 300,
           trace = TRUE, improve = 0.05)
#tra�imo n?jni�i error.rate
which.min(rf.C1$err.rate[,1])

#Izabrali smo 94 drveca
#rf2
rf.C1.2 = randomForest(as.factor(status)~., data=train_data, ntree=300,
                       mtry= 2, importance= TRUE, proximity = TRUE)
print(rf.C1.2)

#testiramo oba modela
#?f model 1
rf.C1.test = predict(rf.C1, newdata=test_data, type="response")
table(rf.C1.test, test_data$status)
df<-data.frame(Orig=test_data$status, Pred= rf.C1.test)
confusionMatrix(table(df$Orig,df$Pred))

#rf model 2
rf.C1.test1 = predict(rf.C1.2, newdat?=test_data, type="response")
table(rf.C1.test1, test_data$status)
df2<-data.frame(Orig=test_data$status, Pred= rf.C1.test1)
confusionMatrix(table(df2$Orig,df2$Pred))

#vizuelizacija 
plot(margin(rf.C1, test_data$status))
plot(margin(rf.C1.2, test_data$stat?s))
varImpPlot(rf.C1.2)
#Mo�emo zakljuciti da je najbitnija varijabla stepen_spondilolisteze

#GBM
suppressMessages(library(gbm))

grid = expand.grid(.n.trees=seq(100,500, by=200), .interaction.depth=seq(1,4, by=1),
                   .shrinkage=c(.001,.01?.1), .n.minobsinnode=10)
control = trainControl(method="CV", number=10) 
#treniramo na� model 
gbm.C1.train = train(status~., data=train_data, method="gbm",
                     trControl=control, tuneGrid=grid)
head(gbm.C1.train)
train_data$status = ifels?(train_data$status=="Abnormal",0,1)
gbm.C1 = gbm(status~., distribution="bernoulli", data=train_data,
             n.trees=100, interaction.depth=1, shrinkage=0.1)

gbm.C1.test = predict(gbm.C1, newdata=test_data, type="response",
                      n.t?ees=100)

gbm.test = ifelse(gbm.C1.test <0.5,"Abnormal", "Normal")

table(gbm.test, test_data$class)

summary(gbm.C1)
#mo�emo uociti da najveci uticaj ima stepen_spondilolisteze sa 57% varijanse, pa nakon
#nakon njega pelvic_tilt.numeric i radijus_karlice ?a 14%
df3<-data.frame(Orig=test_data$status, Pred= gbm.test)
confusionMatrix(table(df3$Orig,df3$Pred))
#ovaj model je mnogo bolji od random forest, jer nam je specificity znacajnije vece

######NAIVE BAYES
#koristimo isti trening i test set podataka
librar?(psych)
describe(C1)

#vizuelizujemo neke od podataka podatke, da bismo imali neku uop�enu predstavu o tome
#�ta se de�ava u na�em setu
#Visual 1
ggplot(C1, aes(stepen_spondilolisteze, colour = status)) +
  geom_freqpoly(binwidth = 1) + labs(title="Stepen ?pondilolisteze u odnosu na klasu")

#visual 2
c <- ggplot(C1, aes(x=radijus_karlice, fill=status, color=status)) +
  geom_histogram(binwidth = 1) + labs(title="Radijus karlice u odnosu na klasu")
c + theme_bw()

#visual 3
P <- ggplot(C1, aes(x=krsni_ugib, ?ill=status, color=status)) +
  geom_histogram(binwidth = 1) + labs(title="Sakralni ugib u odnosu na klasu")
P + theme_bw()

#visual 4
ggplot(C1, aes(lumbalna_lordoza, colour = status)) +
  geom_freqpoly(binwidth = 1) + labs(title="Lumbalna lordoza u odnosu?na klasu")

#pairs
library(GGally)
ggpairs(C1)

#PRAVLJENJE MODELA
#koristimo isti trening i test set iz prethodnih primera

#Check dimensions of the split
prop.table(table(C1$status)) * 100
prop.table(table(train_data$status)) * 100
prop.table(table(test_?ata$status)) * 100

#pravimo promenljivu x koja sadr�i nezavisne promenjive, koje koristimo za predvidanje
#i y koje sadr�i varijablu koju treba da da dobijemo kao odgovor
x = train_data[,-7]
y = train_data$status

library(e1071) #biblioteka koja sadr�i na?ve bayes funkciju
nb <- naiveBayes(x,y)
nb

#evoluacija napravljenog modela
#predvidanje- testiranje modela
nb_pred <- predict(nb, newdata = test_data[-7])

#konfuziona matrica
confusionMatrix(table(nb_pred, test_data$status))
confusionMatrix(test_data$sta?us, nb_pred)
cm = table(test_data[,7], nb_pred)
cm

#Vizuelizacija uticaja svih faktora na na�u izlaznu kategoricku promenljivu
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
model #po�to varImp funkcija nije htela da radi na pocetno? modelu morala sam novi da napravim
X <- varImp(model)
plot(X)
#mo�e se uociti kao i iz prethodnih metoda (random forest), da najveci uticaj ima degree_spond.

####SVM
library('caret')
str(C1)
#koristimo isti test i trening set

#proveravamo dimezije treni?g i test seta
dim(train_data) 
dim(test_data)

#PRAVLJENJE MODELA
#koristimo trainControl, da bismo mogli da iskoristimo train funkciju pri pravljenju SVM modela
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
#method- metod koji ko?istimo, ponovljena kros-validacija
#number- broj iteracija koje ce obaviti 
#repeats- sadr�i setove koji ce davati pri kros-validaciji
#trainControl vraca listu pona�anja koju prosledujemo modelu, pri treniranju

svm<- train(status ~., data = train_data, m?thod = "svmLinear",
            trControl=trctrl,
            preProcess = c("center", "scale"),
            tuneLength = 10)
#status~. oznacava da koristimo sve faktore iz tabele, pri cemu nam je status promenljiva koju 
#treba da dobijemo na izlazu
#preP?ocess- koristi se za pretprocesuiranje podataka (ali smo ih mi prethodno skalirali pa nam
#nije ta funkcija sad neophodna)
svm

#testiranje modela
svm_pred <- predict(svm, newdata= test_data[,-7])
confusionMatrix(table(svm_pred, test_data$status))

#bildov?nje i na�timavanje svm modela sa pro�irenim vrednostima
grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_Grid <- train(status ~., data = train_data, method = "svmLinear",
                  trControl=trctrl,
      ?           #preProcess = c("center", "scale"),
                  tuneGrid = grid,
                  tuneLength = 10)
svm_Grid
#najveca preciznost na�eg modela je ~0,842 pri vrednosti C= 0,1
plot(svm_Grid)

#sada cemo da predvidimo vrednosti za test set sa ?vim pobolj�anim modelom
test_pred_grid <- predict(svm_Grid, newdata = test_data)
confusionMatrix(table(test_pred_grid, test_data$status))
#ovaj model nije bolji, jer nam je specificnost znacajno opala na racun malog rasta senzitivnosti,
#koja nam uistinu j?ste bitnija, ali ne po svaku cenu
#pritom je accuracy kod ovog modela manja nego kod prvog