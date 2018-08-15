# Boltzmann-Simulated-Annealing
Прототип метода отжига (Больцмановский отжиг)
Параметры 
FT mMaxSteps = 1000000; максимальное количество шагов
FT mTemperature = 1; начальная температура

Код относящийся к методу:
//new point Boltzmann
                FT xrec[n];
                while(mOptions.mSwitchOverType == SwitchOverTypes::S_ANNE_Boltzmann) {
                    std::random_device rd{};
                    std::mt19937 gen{rd()};
                    std::uniform_real_distribution<> udis(0.0, 1.0);
                    for (int i = 0; i < n; i++) {
                       std::normal_distribution<FT> z{x[i], sqrt(mOptions.mTemperature)};
                       xold[i] = z(gen);
                       xold[i] = SGMAX(xold[i], box.mA[i]);
                       xold[i] = SGMIN(xold[i], box.mB[i]);
                    }
                    FT fn = obj->func(xold);
                    if(fn < fcur) { 
                       printf(" < nstep=%d fn=%le fcur%le\n", nstep, fn, fcur);
                       fcur = fn;
                       frec = fn;
                       snowgoose::VecUtils::vecCopy(n, xold, x);
                       snowgoose::VecUtils::vecCopy(n, xold, xrec);
                    } else {
                       FT p = exp(-(fn-fcur)/(mOptions.mTemperature/log(1+nstep)));
                       if(p >= udis(gen)) {
//                          fcur = fn;
//                          frec = fn;
                          snowgoose::VecUtils::vecCopy(n, xold, x);
                          printf(" p<= nstep=%d fn=%le fcur%le\n", nstep, fn, fcur);
                       }
                    }
                    nstep++;
                    if(nstep == mOptions.mMaxSteps) {
                       //mOptions.mSwitchOverType = SwitchOverTypes::LESS;
                       snowgoose::VecUtils::vecCopy(n, xrec, x);
//                       break;
                       return;
                    }
                }
//end new point Boltzmann
